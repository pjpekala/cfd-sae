"""Environment and path resolution for hardware presets."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

HARDWARE_CHOICES: tuple[str, ...] = ("auto", "colab", "desktop", "macbook")


def _is_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _get_torch_module():
    try:
        return importlib.import_module("torch")
    except Exception:  # pragma: no cover
        return None


def _cuda_available() -> bool:
    torch_module = _get_torch_module()
    return bool(torch_module is not None and torch_module.cuda.is_available())


def _mps_available() -> bool:
    torch_module = _get_torch_module()
    if torch_module is None:
        return False

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _detect_device() -> str:
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _sanitize_token(token: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", token).strip("-._")
    return cleaned or fallback


def _default_run_name(stage: str, hardware: str, seed: int | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage_token = _sanitize_token(stage, "run")
    seed_token = f"s{seed}" if seed is not None else "sna"
    return f"{stage_token}-{timestamp}-{hardware}-{seed_token}"


def _resolve_hardware(requested: str, on_colab: bool) -> str:
    selected = requested
    if selected == "auto":
        selected = os.getenv("CFD_SAE_HARDWARE", "auto")

    if selected == "auto":
        if on_colab:
            selected = "colab"
        elif platform.system() == "Darwin":
            selected = "macbook"
        else:
            selected = "desktop"

    if selected not in HARDWARE_CHOICES[1:]:
        raise ValueError(f"Unknown hardware preset: {selected}")

    return selected


@dataclass
class Env:
    hardware: str
    device: str
    root: Path
    data_dir: Path
    ckpt_root: Path
    embed_root: Path
    ckpt_dir: Path
    sae_ckpt_dir: Path
    embed_dir: Path
    run_dir: Path
    run_name: str
    run_name_generated: bool
    on_colab: bool


def get_env(
    hardware: str = "auto",
    run_name: str | None = None,
    stage: str = "run",
    seed: int | None = None,
) -> Env:
    """Resolve hardware/device/paths for the current runtime."""
    on_colab = _is_colab()
    resolved_hardware = _resolve_hardware(hardware, on_colab)

    root = Path(__file__).resolve().parents[1]

    base_override = os.getenv("CFD_SAE_BASE_DIR")
    if base_override:
        base = Path(base_override).expanduser()
    elif resolved_hardware == "colab":
        base = Path("/content/drive/MyDrive/cfd-sae")
    else:
        base = root

    if resolved_hardware in {"colab", "desktop"} and _cuda_available():
        device = "cuda"
    elif resolved_hardware == "macbook" and _mps_available():
        device = "mps"
    else:
        device = _detect_device()

    resolved_run_name = run_name or os.getenv("CFD_SAE_RUN_NAME")
    run_name_generated = not bool(resolved_run_name)
    if run_name_generated:
        resolved_run_name = _default_run_name(
            stage=stage, hardware=resolved_hardware, seed=seed
        )
    resolved_run_name = _sanitize_token(resolved_run_name, "run")

    data_dir = base / "data"
    ckpt_root = base / "checkpoints"
    embed_root = base / "embeddings"
    run_root = base / "runs"

    env = Env(
        hardware=resolved_hardware,
        device=device,
        root=root,
        data_dir=data_dir,
        ckpt_root=ckpt_root,
        embed_root=embed_root,
        ckpt_dir=ckpt_root / resolved_run_name,
        sae_ckpt_dir=ckpt_root / resolved_run_name / "sae",
        embed_dir=embed_root / resolved_run_name,
        run_dir=run_root / resolved_run_name,
        run_name=resolved_run_name,
        run_name_generated=run_name_generated,
        on_colab=on_colab,
    )

    for path in (
        env.data_dir,
        env.ckpt_root,
        env.embed_root,
        env.ckpt_dir,
        env.embed_dir,
        env.run_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return env
