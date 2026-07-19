#!/usr/bin/env python3
"""Phase-3 embedding extraction entrypoint.

Loads a trained MGN (best.pt), runs inference over a split, and writes the
pre-decoder node embeddings h_i (paper's frozen embedding for the SAE) as
chunked .npy files under embeddings/<run>/<split>/.

Per the paper (arXiv:2507.16069 §3.2) the SAE trains on TEST-split embeddings,
so default split is "test".

Run:
    uv run python scripts/extract_embeddings.py --hardware macbook \
        --run-name train-mgn --split test --max-batches 10
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
from src.config import load_hardware_config, write_resolved_config, write_run_metadata
from src.env import get_env
from src.models.mgn import MeshGraphNet, MGNConfig
from src.utils.checkpoint import load_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MGN node embeddings.")
    add_common_args(parser, include_resume=False)
    parser.add_argument(
        "--split", default="test", choices=["train", "valid", "test"],
        help="Split to extract (paper uses test).",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Cap number of examples (smoke)."
    )
    parser.add_argument(
        "--mgn-run", default=None,
        help="Run name that produced the MGN checkpoint (default: this run-name).",
    )
    return parser.parse_args()


def to_tensors(sample, device: str):
    import numpy as np

    def t(arr):
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)

    return (
        t(sample.node_features),
        t(sample.edge_index).to(torch.long),
        t(sample.edge_attr),
    )


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware, run_name=args.run_name, stage="extract", seed=args.seed
    )
    config = load_hardware_config(env.root, env.hardware)
    config_path = write_resolved_config(env, config)
    _ = write_run_metadata(env, vars(args), config, stage="extract_embeddings")

    # Load the MGN checkpoint from the training run.
    mgn_run_name = args.mgn_run or env.run_name
    mgn_ckpt_dir = env.ckpt_root / mgn_run_name
    ckpt = load_latest(mgn_ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No MGN checkpoint in {mgn_ckpt_dir}. Train MGN with --run-name {mgn_run_name} first."
        )

    cfg = MGNConfig(
        hidden_dim=int(config.get("mgn", {}).get("hidden_dim", 128)),
        message_passing_steps=int(config.get("mgn", {}).get("message_passing_steps", 9)),
        node_in_dim=8,
        edge_dim=3,
    )
    model = MeshGraphNet(cfg).to(env.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from src.data.cylinder_flow import build_sample, split_reader

    out_dir = env.embed_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    total_vectors = 0
    with torch.no_grad():
        for ex_i, example in enumerate(split_reader(env.data_dir, args.split)):
            if args.max_examples is not None and ex_i >= args.max_examples:
                break
            n_frames = example["velocity"].shape[0]
            for fr in range(n_frames - 1):
                sample = build_sample(example, frame=fr)
                nf, ei, ea = to_tensors(sample, env.device)
                h = model.node_embeddings(nf, ei, ea)  # [N, hidden]
                # Save [N, hidden] float32, one file per (example, frame).
                out_path = out_dir / f"ex{ex_i:05d}_fr{fr:04d}.npy"
                arr = h.detach().cpu().numpy().astype("float32")
                # Chunked write: numpy save is already chunk-friendly per file.
                import numpy as np

                np.save(out_path, arr)
                written += 1
                total_vectors += arr.shape[0]

    print(f"[extract] split={args.split} files={written} node_vectors={total_vectors}")
    print(f"  out_dir={out_dir}")
    print(f"  config={config_path}")


if __name__ == "__main__":
    main()
