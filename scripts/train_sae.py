#!/usr/bin/env python3
"""Phase-3 SAE training entrypoint (resumable).

Loads the extracted MGN node embeddings (chunked .npy) and trains a sparse
autoencoder per arXiv:2507.16069: Linear->ReLU encoder, Linear decoder with
unit-L2-norm dictionary atoms, loss = recon MSE + lambda*||z||_1, batch=128.

Run:
    uv run python scripts/train_sae.py --hardware macbook \
        --run-name train-mgn --epochs 1 --max-steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.cli import add_common_args
from src.config import (
    assert_resume_config_compatible,
    load_hardware_config,
    write_resolved_config,
    write_run_metadata,
)
from src.env import get_env
from src.models.sae import SAEConfig, SparseAutoencoder
from src.utils.checkpoint import load_latest, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE (phase-3).")
    add_common_args(parser, include_resume=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Step cap (smoke).")
    parser.add_argument(
        "--mgn-run",
        default=None,
        help="Run name whose embeddings to use (default: this run-name).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Which extracted split to train the SAE on.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of embedding files held out for early-stopping validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after this many epochs without val-loss improvement.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum epochs before early-stopping can trigger.",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Disable early-stopping; run fixed --epochs instead.",
    )
    return parser.parse_args()


def load_embedding_paths(embed_dir: Path, split: str) -> list[Path]:
    d = embed_dir / split
    if not d.exists():
        raise FileNotFoundError(
            f"No embeddings at {d}. Run extract_embeddings.py --split {split} first."
        )
    paths = sorted(d.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files in {d}.")
    return paths


def split_train_val(paths: list[Path], val_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministic train/val file split (by file, not rows)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(paths))
    n_val = int(round(len(paths) * val_frac))
    n_val = min(max(n_val, 1), len(paths) - 1)
    val_idx = set(int(i) for i in perm[:n_val])
    train = [p for j, p in enumerate(paths) if j not in val_idx]
    val = [p for j, p in enumerate(paths) if j in val_idx]
    return train, val


@torch.no_grad()
def evaluate_val(
    model: "torch.nn.Module",
    val_paths: list[Path],
    mean_t: "torch.Tensor",
    std_t: "torch.Tensor",
    device: str,
    max_files: int = 50,
) -> float:
    """Mean reconstruction MSE over a (subsampled) validation set."""
    model.eval()
    total, n = 0.0, 0
    for p in val_paths[:max_files]:
        x = torch.as_tensor(np.load(p).astype("float32"), device=device)
        x = (x - mean_t) / std_t
        recon, _ = model(x)
        total += float(((recon - x) ** 2).mean()) * x.shape[0]
        n += x.shape[0]
    model.train()
    return total / max(n, 1)


def compute_embedding_stats(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over the embedding set (for z-score normalization)."""
    import numpy as np

    sum_ = None
    sum_sq = None
    n = 0
    for p in paths:
        arr = np.load(p).astype("float32")
        if sum_ is None:
            sum_ = np.zeros(arr.shape[1], dtype="float64")
            sum_sq = np.zeros(arr.shape[1], dtype="float64")
        sum_ += arr.sum(axis=0)
        sum_sq += (arr.astype("float64") ** 2).sum(axis=0)
        n += arr.shape[0]
    mean = sum_ / n
    std = np.sqrt(np.clip(sum_sq / n - mean**2, 1e-12, None))
    return mean.astype("float32"), std.astype("float32")


def embedding_norm_path(embed_dir: Path) -> Path:
    return embed_dir / "embedding_stats.npz"


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware,
        run_name=args.run_name,
        stage="sae",
        seed=args.seed,
        base_dir=Path(args.base_dir) if args.base_dir else None,
    )
    if args.resume and env.run_name_generated:
        raise ValueError("--resume requires an explicit --run-name.")

    config = load_hardware_config(env.root, env.hardware)
    if args.epochs is not None:
        config.setdefault("sae", {})["epochs"] = args.epochs
    if args.resume:
        assert_resume_config_compatible(env.run_dir, config)

    write_resolved_config(env, config)
    _ = write_run_metadata(env, vars(args), config, stage="train_sae")

    paths = load_embedding_paths(env.embed_dir, args.split)
    train_paths, val_paths = split_train_val(paths, args.val_frac, seed=args.seed)
    print(
        f"[sae] embedding files: {len(paths)} (split={args.split}) "
        f"train={len(train_paths)} val={len(val_paths)}"
    )

    # Determine input dim from the first embedding.
    first = np.load(train_paths[0])
    input_dim = int(first.shape[1])

    # Embedding normalization: per-feature z-score, stats computed over the
    # TRAIN subset and persisted so analysis/val reuse the identical transform.
    norm_p = embedding_norm_path(env.embed_dir)
    if norm_p.exists():
        blob = np.load(norm_p)
        emb_mean = blob["mean"].astype("float32")
        emb_std = blob["std"].astype("float32")
        print(f"[sae] reused embedding norm: {norm_p.name}")
    else:
        emb_mean, emb_std = compute_embedding_stats(train_paths)
        np.savez(norm_p, mean=emb_mean, std=emb_std)
        print(f"[sae] computed embedding norm over {len(train_paths)} train files -> {norm_p.name}")

    sae_cfg = SAEConfig(
        input_dim=input_dim,
        expansion=int(config.get("sae", {}).get("expansion", 8)),
        lambda_l1=float(config.get("sae", {}).get("lambda_l1", 3.0e-4)),
    )
    model = SparseAutoencoder(sae_cfg).to(env.device)
    lr = float(config.get("sae", {}).get("lr", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_step = 0
    best_loss = float("inf")
    sae_ckpt_dir = env.sae_ckpt_dir
    sae_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        ckpt = load_latest(sae_ckpt_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("global_step", 0))
            best_loss = float(ckpt.get("best_loss", float("inf")))
            print(f"[resume] global_step={start_step} best_loss={best_loss:.6f}")

    batch_size = int(config.get("sae", {}).get("batch_size", 128))
    model.train()
    global_step = start_step
    epoch = 0

    # Stream embeddings in shuffled mini-batches; cap steps for smoke runs.
    n_paths = len(train_paths)
    # Preload stats to torch tensors for fast normalization on-device.
    mean_t = torch.as_tensor(emb_mean, dtype=torch.float32, device=env.device)
    std_t = torch.as_tensor(emb_std, dtype=torch.float32, device=env.device)

    # Early-stopping bookkeeping (paper: stop when val recon-loss plateaus).
    best_val = float("inf")
    best_val_step = start_step
    epochs_since_improve = 0
    stopped_early = False
    max_epochs = args.epochs or config.get("sae", {}).get("epochs", 1)

    def save_sae(is_best: bool, tag: str | None = None) -> None:
        ck = {
            "global_step": global_step,
            "epoch": epoch,
            "best_loss": best_loss,
            "best_val": best_val,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng": None,
        }
        save_checkpoint(ck, sae_ckpt_dir, epoch=epoch, is_best=is_best)
        if tag is not None:
            torch.save(ck, sae_ckpt_dir / f"{tag}.pt")

    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(n_paths).tolist()
        buffer: list[np.ndarray] = []
        file_idx = 0
        while file_idx < n_paths:
            # Load a few files into the buffer to form batches.
            while len(buffer) < batch_size and file_idx < n_paths:
                arr = np.load(train_paths[perm[file_idx]])
                buffer.append(arr)
                file_idx += 1
            if not buffer:
                break
            # Cap rows per batch by concatenating and truncating to batch_size.
            data = np.concatenate(buffer, axis=0)
            if data.shape[0] > batch_size:
                data = data[:batch_size]
            buffer = []
            x = torch.as_tensor(data, dtype=torch.float32, device=env.device)
            if x.shape[0] == 0:
                continue
            # Per-feature z-score normalization (stable SAE training).
            x = (x - mean_t) / std_t

            optimizer.zero_grad()
            recon, z = model(x)
            loss = model.loss(x)
            if torch.isnan(loss):
                print("[WARN] NaN loss; skipping step")
                continue
            loss.backward()
            optimizer.step()
            # Paper constraint: re-normalize decoder rows to unit L2 after each step.
            model.normalize_decoder()

            global_step += 1
            if global_step % 20 == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            is_best = loss.item() < best_loss
            best_loss = min(best_loss, loss.item())
            if global_step % 200 == 0:
                save_sae(is_best)

            if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
                break
        if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
            break

        # One validation eval per epoch drives early-stopping (paper: stop when
        # held-out recon-loss plateaus).
        if not args.no_val and val_paths:
            val_mse = evaluate_val(model, val_paths, mean_t, std_t, env.device)
            print(f"epoch={epoch} val_mse={val_mse:.6f}")
            if val_mse < best_val:
                best_val = val_mse
                best_val_step = global_step
                epochs_since_improve = 0
                save_sae(is_best=True, tag="best_val")
            else:
                epochs_since_improve += 1

        if not args.no_val and epoch >= args.min_epochs and epochs_since_improve >= args.patience:
            print(
                f"[sae] early stop: no val improvement for {epochs_since_improve} epochs "
                f"(best_val={best_val:.6f} @ step {best_val_step})"
            )
            stopped_early = True
            break

    # Restore best-validation weights (paper-faithful: use plateau model).
    best_val_path = sae_ckpt_dir / "best_val.pt"
    if not args.no_val and best_val_path.exists():
        best_ckpt = torch.load(best_val_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state"])
        print(f"[sae] restored best-val weights (val_mse={best_val:.6f} @ step {best_val_step})")
    save_sae(is_best=True)
    print(
        f"[sae] done. steps={global_step} best_train_loss={best_loss:.6f} "
        f"best_val={best_val:.6f} early_stop={stopped_early}"
    )


if __name__ == "__main__":
    main()
