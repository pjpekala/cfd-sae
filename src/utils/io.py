"""File IO helpers for reproducible run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_json(path: Path, data: Any) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True)
    _atomic_write_text(path, payload + "\n")


def write_yaml(path: Path, data: Any) -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to write yaml. Install dependencies with `uv sync`."
        )
    payload = cast(Any, yaml).safe_dump(data, sort_keys=True)
    _atomic_write_text(path, payload)


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read yaml. Install dependencies with `uv sync`."
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> Any:
    _require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        loaded = cast(Any, yaml).safe_load(handle)
        return loaded or {}
