#!/usr/bin/env python3
"""Phase-3 MGN training entrypoint (resumable).

Trains the MeshGraphNet on next-frame prediction over a split, with periodic
+ best checkpointing and --resume support (restores model + optimizer +
global_step and continues). Single-graph (single-example) batches because node
count N varies per example/split.

Run:
    uv run python scripts/train_mgn.py --hardware macbook --run-name train-mgn \
        --epochs 1
    uv run python scripts/train_mgn.py --hardware macbook --run-name train-mgn \
        --resume
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
from src.utils.checkpoint import load_latest, save_checkpoint
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MGN (phase-3).")
    add_common_args(parser, include_resume=True)
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap (smoke).")
    parser.add_argument(
        "--save-every", type=int, default=200, help="Checkpoint every N global steps."
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


def build_sample_dataset(data_dir: Path, split: str):
    """Yield (frame) GraphSamples across all examples/frames of a split."""
    from src.data.cylinder_flow import build_sample, split_reader

    for example in split_reader(data_dir, split):
        n_frames = example["velocity"].shape[0]
        for fr in range(n_frames - 1):
            yield build_sample(example, frame=fr)


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware,
        run_name=args.run_name,
        stage="mgn",
        seed=args.seed,
        base_dir=Path(args.base_dir) if args.base_dir else None,
    )
    if args.resume and env.run_name_generated:
        raise ValueError("--resume requires an explicit --run-name.")

    config = load_hardware_config(env.root, env.hardware)
    if args.epochs is not None:
        config.setdefault("mgn", {})["epochs"] = args.epochs

    if args.resume:
        assert_resume_config_compatible(env.run_dir, config)

    set_seed(args.seed)

    write_resolved_config(env, config)
    _ = write_run_metadata(env, vars(args), config, stage="train_mgn")

    cfg = MGNConfig(
        hidden_dim=int(config.get("mgn", {}).get("hidden_dim", 128)),
        message_passing_steps=int(config.get("mgn", {}).get("message_passing_steps", 9)),
        node_in_dim=8,
        edge_dim=3,
    )
    model = MeshGraphNet(cfg).to(env.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = load_latest(env.ckpt_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("global_step", 0))
            best_loss = float(ckpt.get("best_loss", float("inf")))
            print(f"[resume] restored global_step={start_step} best_loss={best_loss:.6f}")

    print(f"[train] MGN params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"hardware={env.hardware} device={env.device} run_name={env.run_name}")

    dataset = build_sample_dataset(env.data_dir, "train")
    model.train()
    global_step = start_step
    nan_seen = False
    epoch = 0

    for epoch in range(1, (args.epochs or config.get("mgn", {}).get("epochs", 1)) + 1):
        for sample in dataset:
            if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
                break
            nf, ei, ea, tv, tp = to_tensors(sample, env.device)
            optimizer.zero_grad()
            pred_vel, pred_pres = model(nf, ei, ea)
            loss = model.loss(pred_vel, pred_pres, tv, tp)
            if torch.isnan(loss):
                nan_seen = True
                print("[WARN] NaN loss; skipping step")
                continue
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                is_best = True
            else:
                is_best = False

            if global_step % args.save_every == 0:
                ckpt = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "rng": None,
                }
                save_checkpoint(ckpt, env.ckpt_dir, epoch=epoch, is_best=is_best)
        if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
            break

    # Final checkpoint.
    ckpt = {
        "global_step": global_step,
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng": None,
    }
    ckpt_path = save_checkpoint(ckpt, env.ckpt_dir, epoch=epoch, is_best=True)
    print(f"[train] done. steps={global_step} best_loss={best_loss:.6f} nan={nan_seen}")
    print(f"  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
