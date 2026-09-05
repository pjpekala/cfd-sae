#!/usr/bin/env python3
"""Phase-3 MGN training entrypoint (resumable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="""Maximum global steps to train. Training stops when global_step
        reaches this value. Set to ~400000 to match the paper's typical stopping
        point where loss plateaus. When combined with --patience, also enables
        epoch-based early stopping if validation loss doesn't improve.""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size. If None, uses config default (64 for "
        "CylinderFlow, 32 for 3D C3FD).",
    )
    parser.add_argument(
        "--save-every", type=int, default=200, help="Checkpoint every N global steps."
    )
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise std on input velocity (normalized space). 0 to disable.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable feature normalization."
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Disable per-epoch shuffling.")
    return parser.parse_args()


def _tb_writer(tb_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    try:
        return SummaryWriter(str(tb_dir))
    except Exception:
        return None


def to_tensors(sample, device: str):
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
        base_dir=Path(args.base_dir) if args.base_dir else None,
    )
    if args.resume and env.run_name_generated:
        raise ValueError("--resume requires an explicit --run-name.")

    config = load_hardware_config(env.root, env.hardware)
    if args.epochs is not None:
        config.setdefault("mgn", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("mgn", {})["batch_size"] = args.batch_size

    if args.resume:
        assert_resume_config_compatible(env.run_dir, config)

    set_seed(args.seed)

    write_resolved_config(env, config)
    _ = write_run_metadata(env, vars(args), config, stage="train_mgn")

    cfg = MGNConfig(
        hidden_dim=int(config.get("mgn", {}).get("hidden_dim", 128)),
        message_passing_steps=15,
        node_in_dim=8,
        edge_dim=3,
    )
    model = MeshGraphNet(cfg).to(env.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_step = 0
    start_epoch = 1
    start_sample_idx = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = load_latest(env.ckpt_dir)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("global_step", 0))
            best_loss = float(ckpt.get("best_loss", float("inf")))
            start_epoch = int(ckpt.get("epoch", 1))
            start_sample_idx = int(ckpt.get("sample_idx", 0))
            # next epoch if previous epoch was fully completed
            # sample_idx == 0 or < len means resume mid-epoch; otherwise advance
            print(
                f"[resume] step={start_step} epoch={start_epoch} "
                f"sample_idx={start_sample_idx} best={best_loss:.6f}"
            )

    print(f"[train] MGN params: {sum(p.numel() for p in model.parameters()):,}")
    batch_size = args.batch_size or config.get("mgn", {}).get("batch_size", 64)
    print(f"[train] batch_size={batch_size} (per-GPU)")
    print(f"hardware={env.hardware} device={env.device} run_name={env.run_name}")

    stats = None
    if not args.no_normalize:
        from src.data.cylinder_flow import compute_stats, load_stats, save_stats, stats_path

        sp = stats_path(env.data_dir)
        if sp.exists():
            stats = load_stats(env.data_dir)
            print(f"[normalize] loaded stats from {sp}")
        else:
            print("[normalize] computing stats over train split...")
            stats = compute_stats(env.data_dir, "train")
            save_stats(stats, env.data_dir)
            print(f"[normalize] saved stats to {sp}")
        print(f"  velocity mean={stats.velocity_mean} std={stats.velocity_std}")
        print(f"  pressure mean={stats.pressure_mean:.4f} std={stats.pressure_std:.4f}")
        print(f"  edge_rel mean={stats.edge_rel_mean} std={stats.edge_rel_std}")
        print(f"  edge_dist mean={stats.edge_dist_mean:.4f} std={stats.edge_dist_std:.4f}")

    tb_writer = None if args.no_tb else _tb_writer(env.tb_dir)
    if tb_writer is not None:
        print(f"[tb] logging to {env.tb_dir}")
    elif not args.no_tb:
        print("[tb] tensorboard not available; logging disabled")

    from src.data.cylinder_flow import build_sample, normalize_graph_sample, split_reader

    print("[data] loading train examples into memory...")
    examples = list(split_reader(env.data_dir, "train"))
    if not examples:
        raise RuntimeError("No training examples found")
    flat_index: list[tuple[int, int]] = []
    for ei, ex in enumerate(examples):
        n_frames = int(ex["velocity"].shape[0])
        for fr in range(n_frames - 1):
            flat_index.append((ei, fr))
    print(f"[data] {len(examples)} examples, {len(flat_index)} frame samples")

    model.train()
    global_step = start_step
    nan_seen = False
    num_epochs = args.epochs or config.get("mgn", {}).get("epochs", 1)
    epoch = start_step = 0  # reset
    # If resuming mid-epoch and start_sample_idx >= len(flat_index), advance to next epoch
    if start_sample_idx >= len(flat_index):
        start_epoch += 1
        start_sample_idx = 0

    # Set up LR scheduler: exponential decay from 1e-3 to 1e-7.
    # If max-steps provided, decay over that many steps.
    # Otherwise decay over estimated full training duration.
    max_steps_arg = args.max_steps
    if max_steps_arg is None:
        max_steps_arg = num_epochs * len(flat_index)
    lr_final = 1e-7
    lr_init = 1e-3
    gamma = (lr_final / lr_init) ** (1.0 / max_steps_arg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(start_epoch, num_epochs + 1):
        is_resumed_epoch = epoch == start_epoch and start_sample_idx > 0
        indices = list(range(len(flat_index)))
        if not args.no_shuffle:
            rng = np.random.default_rng((args.seed or 0) + epoch)
            rng.shuffle(indices)
        else:
            # deterministic order
            pass

        start_idx = start_sample_idx if is_resumed_epoch else 0
        # reset for next epochs
        start_sample_idx = 0

        if start_idx > 0:
            print(f"[resume] epoch {epoch}: skipping {start_idx}/{len(indices)} samples")

        for pos in range(start_idx, len(indices)):
            if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
                break
            ei, fr = flat_index[indices[pos]]
            sample = build_sample(examples[ei], frame=fr)
            if stats is not None:
                sample = normalize_graph_sample(sample, stats)

            nf, ei_t, ea, tv, tp = to_tensors(sample, env.device)

            if args.noise_std and args.noise_std > 0 and stats is not None:
                noise = (
                    torch.randn(nf.shape[0], 2, device=nf.device, dtype=nf.dtype) * args.noise_std
                )
                nf[:, velocity_offset : velocity_offset + 2] = (
                    nf[:, velocity_offset : velocity_offset + 2] + noise
                )

            optimizer.zero_grad()
            pred_vel, pred_pres = model(nf, ei_t, ea)
            loss = model.loss(pred_vel, pred_pres, tv, tp)
            if torch.isnan(loss):
                nan_seen = True
                print("[WARN] NaN loss; skipping step")
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            sample_idx_next = pos + 1

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss.item(), global_step)
                if global_step % 20 == 0:
                    tb_writer.flush()
            if global_step % 20 == 0:
                msg = f"epoch={epoch} step={global_step} pos={pos + 1}/{len(indices)}"
                print(f"{msg} loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                is_best = True
            else:
                is_best = False

            if global_step % args.save_every == 0:
                at_end = sample_idx_next >= len(indices)
                ckpt = {
                    "global_step": global_step,
                    "epoch": epoch + 1 if at_end else epoch,
                    "sample_idx": 0 if at_end else sample_idx_next,
                    "best_loss": best_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "rng": None,
                }
                save_checkpoint(ckpt, env.ckpt_dir, epoch=epoch, is_best=is_best)
        if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
            break
        # after completing an epoch, ensure sample_idx reset
        if args.max_steps is not None and (global_step - start_step) >= args.max_steps:
            break

    ckpt = {
        "global_step": global_step,
        "epoch": epoch,
        "sample_idx": 0,
        "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng": None,
    }
    ckpt_path = save_checkpoint(ckpt, env.ckpt_dir, epoch=epoch, is_best=True)
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
        print(f"[tb] closed {env.tb_dir}")
    print(f"[train] done. steps={global_step} best_loss={best_loss:.6f} nan={nan_seen}")
    print(f"  checkpoint={ckpt_path}")
    print(f"  tb_logs={env.tb_dir}")


if __name__ == "__main__":
    main()
