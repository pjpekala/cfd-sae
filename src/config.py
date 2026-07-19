"""Configuration loading and run metadata helpers."""

from __future__ import annotations

import copy
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from src.env import Env
from src.utils.io import write_json, write_yaml


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configs. Install dependencies with `uv sync`."
        )


def _git_stdout(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None

    value = result.stdout.strip()
    return value or None


def git_metadata(root: Path) -> dict[str, Any]:
    commit = _git_stdout(root, ["rev-parse", "HEAD"])
    dirty_status = _git_stdout(root, ["status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(dirty_status) if dirty_status is not None else None,
    }


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_hardware_config(
    root: Path, hardware: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    _require_yaml()

    path = root / "configs" / "hardware" / f"{hardware}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing hardware config: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = cast(Any, yaml).safe_load(handle)
        config = loaded or {}

    if overrides:
        config = deep_merge(config, overrides)

    return config


def write_resolved_config(env: Env, config: dict[str, Any]) -> Path:
    output_path = env.run_dir / "resolved_config.yaml"
    write_yaml(output_path, config)
    return output_path


def write_run_metadata(
    env: Env, args: dict[str, Any], config: dict[str, Any], stage: str
) -> Path:
    git = git_metadata(env.root)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "git": git,
        "env": {
            "hardware": env.hardware,
            "device": env.device,
            "root": str(env.root),
            "data_dir": str(env.data_dir),
            "ckpt_dir": str(env.ckpt_dir),
            "embed_dir": str(env.embed_dir),
            "run_dir": str(env.run_dir),
            "run_name": env.run_name,
            "on_colab": env.on_colab,
        },
        "args": args,
        "config": config,
    }
    output_path = env.run_dir / "run_metadata.json"
    write_json(output_path, metadata)
    return output_path


def read_yaml(path: Path) -> dict[str, Any]:
    _require_yaml()

    with path.open("r", encoding="utf-8") as handle:
        loaded = cast(Any, yaml).safe_load(handle)
        data = loaded or {}
    return data


def assert_resume_config_compatible(
    run_dir: Path,
    current_config: dict[str, Any],
    critical_keys: tuple[str, ...] = ("mgn", "sae", "batch_size"),
) -> None:
    snapshot = run_dir / "resolved_config.yaml"
    if not snapshot.exists():
        return

    previous = read_yaml(snapshot)
    mismatches: list[str] = []
    for key in critical_keys:
        if previous.get(key) != current_config.get(key):
            mismatches.append(key)

    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            "Resume config mismatch for keys: "
            f"{joined}. Use a new --run-name for incompatible changes."
        )
