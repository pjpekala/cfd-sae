#!/usr/bin/env python3
"""Phase-2 smoke training entrypoint for MeshGraphNet.

Trains a real (if tiny) MGN on cylinder-flow next-frame prediction. This is a
smoke loop: single-example (single-graph) batches over a capped number of
frames/steps, with checkpointing and run metadata. It proves the data -> model
-> loss -> backward path works end to end on real data; it does NOT aim for
convergence.

Run:
    uv run python scripts/train_mgn.py --hardware macbook --run-name smoke-mgn \
        --epochs 1 --max-steps 5
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
from src.models.mgn import MeshGraphNet, MGNConfig
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MGN (phase-2 smoke).")
    add_common_args(parser, include_resume=True)
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--max-steps", type=int, default=5, help="Max optimizer steps.")
    parser.add_argument(
        "--frames-per-example", type=int, default=4,
        help="Frames sampled per example for the smoke loop.",
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
        t(sample.target_velocity),
        t(sample.target_pressure),
    )


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware,
        run_name=args.run_name,
        stage="mgn",
        seed=args.seed,
    )

    if args.resume and env.run_name_generated:
        raise ValueError("--resume requires an explicit --run-name or CFD_SAE_RUN_NAME.")

    config = load_hardware_config(env.root, env.hardware)
    if args.epochs is not None:
        config.setdefault("mgn", {})["epochs"] = args.epochs

    if args.resume:
        assert_resume_config_compatible(env.run_dir, config)

    set_seed(args.seed)

    config_path = write_resolved_config(env, config)
    metadata_path = write_run_metadata(env, vars(args), config, stage="train_mgn")

    from src.data.cylinder_flow import build_sample, split_reader

    cfg = MGNConfig(
        hidden_dim=int(config.get("mgn", {}).get("hidden_dim", 128)),
        message_passing_steps=int(config.get("mgn", {}).get("message_passing_steps", 9)),
        node_in_dim=8,
        edge_dim=3,
    )
    model = MeshGraphNet(cfg).to(env.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"[phase-2] MGN model built: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"hardware={env.hardware} device={env.device} run_name={env.run_name}")
    print(f"config={config_path}")
    print(f"metadata={metadata_path}")

    model.train()
    global_step = 0
    total_loss = 0.0
    nan_seen = False

    for ex_i, example in enumerate(split_reader(env.data_dir, "valid", max_examples=2)):
        n_frames = example["velocity"].shape[0]
        frames = list(range(0, min(args.frames_per_example, n_frames - 1)))
        for fr in frames:
            if global_step >= args.max_steps:
                break
            sample = build_sample(example, frame=fr)
            nf, ei, ea, tv, tp = to_tensors(sample, env.device)

            optimizer.zero_grad()
            pred_vel, pred_pres = model(nf, ei, ea)
            loss = model.loss(pred_vel, pred_pres, tv, tp)
            if torch.isnan(loss):
                nan_seen = True
                print(f"[WARN] NaN loss at example {ex_i} frame {fr}")
                continue
            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += float(loss.item())
            print(f"step {global_step:03d} | ex={ex_i} fr={fr} | loss={loss.item():.6f}")

        if global_step >= args.max_steps:
            break

    if global_step == 0:
        raise RuntimeError("No training steps executed; check data path / max_steps.")

    # Save a smoke checkpoint (epoch 0; current global_step).
    ckpt = {
        "global_step": global_step,
        "epoch": 0,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng": None,
    }
    ckpt_path = save_checkpoint(ckpt, env.ckpt_dir, epoch=0, is_best=False, keep_last=3)

    print("[phase-2] Smoke training complete.")
    print(f"  steps={global_step} mean_loss={total_loss / max(global_step, 1):.6f}")
    print(f"  nan_seen={nan_seen}")
    print(f"  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
