#!/usr/bin/env python3
"""Phase-3 SAE analysis entrypoint.

Loads a trained SAE + extracted embeddings and computes interpretability
diagnostics per arXiv:2507.16069:
  - reconstruction MSE, mean L1 of codes (sparsity proxy)
  - per-latent Top-K saliency using the three Table-1 scores:
      Variance   s_var(d) = var over samples of z_{i,d}
      MeanAbs    s_abs(d) = mean |z_{i,d}|
      Entropy    s_ent(d) = -sum_b p_{b,d} log p_{b,d} over B histogram bins
Results are written to runs/<run>/analysis.json.

Run:
    uv run python scripts/analyze.py --hardware macbook --run-name train-mgn \
        --split test --top-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.cli import add_common_args
from src.config import load_hardware_config, write_resolved_config, write_run_metadata
from src.env import get_env
from src.models.sae import SAEConfig, SparseAutoencoder
from src.utils.checkpoint import load_latest
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAE outputs.")
    add_common_args(parser, include_resume=False)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--entropy-bins", type=int, default=50)
    parser.add_argument("--mgn-run", default=None, help="Run name whose embeddings/SAnE to use.")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of embedding files to process per batch (reduces RAM usage). Defaults to hardware config analysis_batch_size (500 for desktop/colab/macbook).")
    return parser.parse_args()


def load_normalizer(embed_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    p = embed_dir / "embedding_stats.npz"
    if not p.exists():
        return None
    blob = np.load(p)
    return blob["mean"].astype("float32"), blob["std"].astype("float32")


def normalize(arr: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if stats is None:
        return arr
    mean, std = stats
    return (arr - mean) / std


def load_all_codes_batch(
    embed_dir: Path, split: str, sae_run_dir: Path, batch_size: int
) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Yield codes in batches to reduce RAM usage.

    Returns list of code batches (each [N_d, hidden]) and normalization stats.
    """
    d = embed_dir / split
    if not d.exists():
        raise FileNotFoundError(f"No embeddings at {d}.")
    paths = sorted(d.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy in {d}.")

    # Need SAE to encode embeddings -> codes. Load SAE checkpoint.
    ckpt = load_latest(sae_run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No SAE checkpoint in {sae_run_dir}.")
    first = np.load(paths[0])
    sae = SparseAutoencoder(SAEConfig(input_dim=int(first.shape[1])))
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    stats = load_normalizer(env.embed_dir)

    import torch

    code_batches: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            arrs = [np.load(p).astype("float32") for p in batch_paths]
            x = torch.as_tensor(np.concatenate(arrs, axis=0))
            if stats is not None:
                mean_t = torch.as_tensor(stats[0].astype("float32"))
                std_t = torch.as_tensor(stats[1].astype("float32"))
                x = (x - mean_t) / std_t
            z = sae.encode(x)
            code_batches.append(z.numpy())
    return code_batches, stats


def compute_salient_scores_incremental(
    code_batches: list[np.ndarray], entropy_bins: int
) -> dict[str, np.ndarray]:
    """Compute Table-1 scores incrementally from code batches.

    Avoids loading all [M, hidden] codes at once.
    """
    n_latents = code_batches[0].shape[1]
    n_samples = 0

    # Running sums for variance and mean_abs (Welford-like)
    sum_z = np.zeros(n_latents, dtype=np.float64)
    sum_z_sq = np.zeros(n_latents, dtype=np.float64)
    sum_abs = np.zeros(n_latents, dtype=np.float64)

    # Incremental histograms for entropy
    hist_accum = np.zeros((n_latents, entropy_bins), dtype=np.int64)

    for z_batch in code_batches:
        M_b = z_batch.shape[0]
        n_samples += M_b

        # Variance accumulation: maintain sum and sum_sq
        sum_z += z_batch.sum(axis=0)
        sum_z_sq += (z_batch ** 2).sum(axis=0)
        sum_abs += np.abs(z_batch).sum(axis=0)

        # Histogram per latent dimension
        for d in range(n_latents):
            hist, _ = np.histogram(z_batch[:, d], bins=entropy_bins)
            hist_accum[d] += hist

    # Finalize statistics
    variance = sum_z_sq / n_samples - (sum_z / n_samples) ** 2
    mean_abs = sum_abs / n_samples

    # Entropy from accumulated histograms
    ent = np.zeros(n_latents, dtype=np.float64)
    for d in range(n_latents):
        p = hist_accum[d].astype(np.float64) / n_samples
        p = p[p > 0]
        ent[d] = -(p * np.log(p)).sum()

    return {"variance": variance, "mean_abs": mean_abs, "entropy": ent}


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware,
        run_name=args.run_name,
        stage="analysis",
        seed=args.seed,
        base_dir=Path(args.base_dir) if args.base_dir else None,
    )
    config = load_hardware_config(env.root, env.hardware)
    # Use hardware-configurable batch size for analysis to reduce RAM usage
    if args.batch_size is None:
        args.batch_size = config.get("analysis_batch_size", 500)
    write_resolved_config(env, config)
    _ = write_run_metadata(env, vars(args), config, stage="analyze")

    # SAE checkpoint lives under the same run-name, in the sae subdir.
    sae_run_dir = env.sae_ckpt_dir
    ckpt = load_latest(sae_run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No SAE checkpoint in {sae_run_dir}. Train SAE first.")

    first_path = sorted((env.embed_dir / args.split).glob("*.npy"))
    if not first_path:
        raise FileNotFoundError(f"No embeddings at {env.embed_dir / args.split}.")
    input_dim = int(np.load(first_path[0]).shape[1])
    sae = SparseAutoencoder(SAEConfig(input_dim=input_dim))
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    stats = load_normalizer(env.embed_dir)

    import torch

    # Single-pass processing: load each embedding, encode once, accumulate stats.
    # This avoids storing all [M, hidden] codes in memory (Option C).
    all_paths = sorted((env.embed_dir / args.split).glob("*.npy"))
    n_samples = 0
    total_mse = 0.0
    total_l1 = 0.0
    total_inactive = 0
    # Running sums for variance / mean_abs (incremental Welford-like)
    sum_z = None
    sum_z_sq = None
    sum_abs = None
    # Running histogram for entropy
    hist_accum = None
    hidden = 0

    with torch.no_grad():
        for i in range(0, len(all_paths), args.batch_size):
            batch_paths = all_paths[i : i + args.batch_size]
            # Load and normalize embeddings for this batch
            arrs = [np.load(p).astype("float32") for p in batch_paths]
            x_np = np.concatenate(arrs, axis=0)
            x_t = torch.as_tensor(normalize(x_np, stats))

            # Encode through SAE
            recon, z_t = sae(x_t)

            M_b = x_t.shape[0]
            n_samples += M_b
            hidden = x_t.shape[1] if hidden == 0 else hidden

            # Accumulate reconstruction MSE
            batch_mse = float(((recon - x_t) ** 2).mean())
            total_mse += batch_mse * M_b

            # Accumulate mean L1 codes
            total_l1 += float(z_t.abs().mean()) * M_b

            # Accumulate fraction of near-zero activations (sparsity)
            total_inactive += float((z_t.abs() < 1e-3).float().mean()) * M_b

            # Incremental variance / mean_abs accumulation (Welford-like)
            if sum_z is None:
                sum_z = z_t.sum(axis=0).cpu().numpy().astype(np.float64)
                sum_z_sq = (z_t ** 2).sum(axis=0).cpu().numpy().astype(np.float64)
                sum_abs = np.abs(z_t).sum(axis=0).cpu().numpy().astype(np.float64)
                # Histogram init: [n_latents, entropy_bins]
                hist_accum = np.zeros((hidden, args.entropy_bins), dtype=np.int64)
            else:
                sum_z += z_t.sum(axis=0).cpu().numpy().astype(np.float64)
                sum_z_sq += (z_t ** 2).sum(axis=0).cpu().numpy().astype(np.float64)
                sum_abs += np.abs(z_t).sum(axis=0).cpu().numpy().astype(np.float64)

            # Accumulate per-latent histograms for entropy
            for d in range(hidden):
                hist, _ = np.histogram(z_t[:, d].cpu().numpy(), bins=args.entropy_bins)
                hist_accum[d] += hist

            # Free memory between batches
            del arrs, x_np, x_t, recon, z_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Finalize statistics
    recon_mse = total_mse / n_samples if n_samples > 0 else 0.0
    l1_mean = total_l1 / n_samples if n_samples > 0 else 0.0
    frac_inactive = total_inactive / n_samples if n_samples > 0 else 0.0

    # Variance = E[z^2] - (E[z])^2
    variance = sum_z_sq / n_samples - (sum_z / n_samples) ** 2 if sum_z is not None else np.zeros(hidden)
    mean_abs = sum_abs / n_samples if sum_abs is not None else np.zeros(hidden)

    # Entropy from accumulated histograms
    ent = np.zeros(hidden, dtype=np.float64)
    for d in range(hidden):
        p = hist_accum[d].astype(np.float64) / n_samples
        p = p[p > 0]
        ent[d] = -(p * np.log(p)).sum()

    scores = {"variance": variance, "mean_abs": mean_abs, "entropy": ent}

    top_k_idx: dict[str, list[int]] = {}
    for name, s in scores.items():
        order = np.argsort(-s)[: args.top_k]
        top_k_idx[name] = [int(i) for i in order]

    results = {
        "split": args.split,
        "n_samples": int(n_samples),
        "hidden_dim": int(hidden),
        "recon_mse": recon_mse,
        "mean_l1_codes": l1_mean,
        "frac_inactive_codes": frac_inactive,
        "top_k": args.top_k,
        "top_k_latents": top_k_idx,
        "score_variance_topk": {
            str(k): float(scores["variance"][k]) for k in top_k_idx["variance"]
        },
        "score_meanabs_topk": {str(k): float(scores["mean_abs"][k]) for k in top_k_idx["mean_abs"]},
        "score_entropy_topk": {str(k): float(scores["entropy"][k]) for k in top_k_idx["entropy"]},
    }
    out_path = env.run_dir / "analysis.json"
    write_json(out_path, results)

    print(f"[analyze] split={args.split} samples={n_samples} hidden={hidden}")
    print(f"  recon_mse={recon_mse:.6f} mean_l1={l1_mean:.6f} frac_inactive={frac_inactive:.3f}")
    print(f"  Top-{args.top_k} by Variance: {top_k_idx['variance'][:10]} ...")
    print(f"  Top-{args.top_k} by MeanAbs:  {top_k_idx['mean_abs'][:10]} ...")
    print(f"  results={out_path}")


if __name__ == "__main__":
    main()
