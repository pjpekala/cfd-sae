"""Checkpoint helpers with retention for resumable runs."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    if torch is None:
        raise RuntimeError("torch is required for checkpoint saving")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, dir=path.parent, suffix=".pt") as tmp:
        tmp_path = Path(tmp.name)
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def save_checkpoint(
    state: dict[str, Any],
    ckpt_dir: Path,
    epoch: int,
    is_best: bool = False,
    keep_last: int = 3,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / f"epoch_{epoch:04d}.pt"
    _atomic_torch_save(state, target)

    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    stale = checkpoints[:-keep_last] if keep_last > 0 else checkpoints
    for path in stale:
        path.unlink(missing_ok=True)

    if is_best:
        _atomic_torch_save(state, ckpt_dir / "best.pt")

    return target


def load_latest(ckpt_dir: Path) -> dict[str, Any] | None:
    if torch is None:
        raise RuntimeError("torch is required for checkpoint loading")
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    state = torch.load(latest, map_location="cpu")
    return state
