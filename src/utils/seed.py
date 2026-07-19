"""Deterministic seeding helpers."""

from __future__ import annotations

import importlib
import os
import random
from functools import lru_cache


@lru_cache(maxsize=1)
def _numpy_module():
    try:
        return importlib.import_module("numpy")
    except Exception:  # pragma: no cover
        return None


@lru_cache(maxsize=1)
def _torch_module():
    try:
        return importlib.import_module("torch")
    except Exception:  # pragma: no cover
        return None


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    numpy_module = _numpy_module()
    if numpy_module is not None:
        numpy_module.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch_module = _torch_module()
    if torch_module is None:
        return

    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch_module.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.benchmark = False
            torch_module.backends.cudnn.deterministic = True
