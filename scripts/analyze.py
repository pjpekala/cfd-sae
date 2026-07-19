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
    parser.add_argument(
        "--mgn-run", default=None, help="Run name whose embeddings/SAnE to use."
    )
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


def load_all_codes(
    embed_dir: Path, split: str, sae_run_dir: Path
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
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

    stats = load_normalizer(embed_dir)
    import torch

    codes_chunks = []
    with torch.no_grad():
        for p in paths:
            arr = np.load(p).astype("float32")
            x = torch.as_tensor(normalize(arr, stats))
            z = sae.encode(x)
            codes_chunks.append(z.numpy())
    return np.concatenate(codes_chunks, axis=0), stats  # [M, hidden]


def salient_scores(z: np.ndarray, bins: int) -> dict[str, np.ndarray]:
    """Paper Table-1 scores over samples M for each latent dim d."""
    var = z.var(axis=0)
    mean_abs = np.abs(z).mean(axis=0)
    # Entropy per latent over histogram bins (flatten to per-dim histograms).
    ent = np.zeros(z.shape[1], dtype=np.float64)
    for d in range(z.shape[1]):
        hist, _ = np.histogram(z[:, d], bins=bins)
        p = hist / hist.sum()
        p = p[p > 0]
        ent[d] = -(p * np.log(p)).sum()
    return {"variance": var, "mean_abs": mean_abs, "entropy": ent}


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware, run_name=args.run_name, stage="analysis", seed=args.seed
    )
    config = load_hardware_config(env.root, env.hardware)
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

    import torch

    z_all, stats = load_all_codes(env.embed_dir, args.split, sae_run_dir)
    M, hidden = z_all.shape

    # Reconstruction metrics on the (normalized) embeddings the SAE was trained on.
    x_all = np.concatenate(
        [
            normalize(np.load(p).astype("float32"), stats)
            for p in sorted((env.embed_dir / args.split).glob("*.npy"))
        ],
        axis=0,
    )
    with torch.no_grad():
        x_t = torch.as_tensor(x_all)
        recon, z_t = sae(x_t)
        recon_mse = float(((recon - x_t) ** 2).mean())
        l1_mean = float(z_t.abs().mean())
        # Fraction of near-zero activations (sparsity).
        frac_inactive = float((z_t.abs() < 1e-3).float().mean())

    scores = salient_scores(z_all, args.entropy_bins)
    top_k_idx: dict[str, list[int]] = {}
    for name, s in scores.items():
        order = np.argsort(-s)[: args.top_k]
        top_k_idx[name] = [int(i) for i in order]

    results = {
        "split": args.split,
        "n_samples": int(M),
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

    print(f"[analyze] split={args.split} samples={M} hidden={hidden}")
    print(f"  recon_mse={recon_mse:.6f} mean_l1={l1_mean:.6f} "
          f"frac_inactive={frac_inactive:.3f}")
    print(f"  Top-{args.top_k} by Variance: {top_k_idx['variance'][:10]} ...")
    print(f"  Top-{args.top_k} by MeanAbs:  {top_k_idx['mean_abs'][:10]} ...")
    print(f"  results={out_path}")


if __name__ == "__main__":
    main()
