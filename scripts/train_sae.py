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
        "--mgn-run", default=None,
        help="Run name whose embeddings to use (default: this run-name).",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "valid", "test"],
        help="Which extracted split to train the SAE on.",
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


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware, run_name=args.run_name, stage="sae", seed=args.seed
    )
    if args.resume and env.run_name_generated:
        raise ValueError("--resume requires an explicit --run-name or CFD_SAE_RUN_NAME.")

    config = load_hardware_config(env.root, env.hardware)
    if args.epochs is not None:
        config.setdefault("sae", {})["epochs"] = args.epochs
    if args.resume:
        assert_resume_config_compatible(env.run_dir, config)

    write_resolved_config(env, config)
    _ = write_run_metadata(env, vars(args), config, stage="train_sae")

    paths = load_embedding_paths(env.embed_dir, args.split)
    print(f"[sae] embedding files: {len(paths)} (split={args.split})")

    # Determine input dim from the first embedding.
    import numpy as np

    first = np.load(paths[0])
    input_dim = int(first.shape[1])

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
    if args.resume:
        ckpt = load_latest(env.ckpt_dir)
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
    n_paths = len(paths)
    for epoch in range(1, (args.epochs or config.get("sae", {}).get("epochs", 1)) + 1):
        perm = torch.randperm(n_paths).tolist()
        buffer: list[np.ndarray] = []
        file_idx = 0
        step_in_epoch = 0
        while file_idx < n_paths:
            # Load a few files into the buffer to form batches.
            while len(buffer) < batch_size and file_idx < n_paths:
                arr = np.load(paths[perm[file_idx]])
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
            step_in_epoch += 1
            if global_step % 20 == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            is_best = loss.item() < best_loss
            best_loss = min(best_loss, loss.item())
            if global_step % 200 == 0:
                ckpt = {
                    "global_step": global_step, "epoch": epoch, "best_loss": best_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "rng": None,
                }
                save_checkpoint(ckpt, env.ckpt_dir, epoch=epoch, is_best=is_best)

            if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
                break
        if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
            break

    ckpt = {
        "global_step": global_step, "epoch": epoch, "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "rng": None,
    }
    ckpt_path = save_checkpoint(ckpt, env.ckpt_dir, epoch=epoch, is_best=True)
    print(f"[sae] done. steps={global_step} best_loss={best_loss:.6f}")
    print(f"  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
