#!/usr/bin/env python3
"""Phase-1 stub for embedding extraction entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import add_common_args
from src.config import (
    assert_resume_config_compatible,
    load_hardware_config,
    write_resolved_config,
    write_run_metadata,
)
from src.env import get_env
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings (phase-1 scaffold)."
    )
    add_common_args(parser, include_resume=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = get_env(
        hardware=args.hardware,
        run_name=args.run_name,
        stage="extract",
        seed=args.seed,
    )

    if args.resume and env.run_name_generated:
        raise ValueError(
            "--resume requires an explicit --run-name or CFD_SAE_RUN_NAME."
        )

    config = load_hardware_config(env.root, env.hardware)

    if args.resume:
        assert_resume_config_compatible(env.run_dir, config)

    set_seed(args.seed)

    config_path = write_resolved_config(env, config)
    metadata_path = write_run_metadata(
        env, vars(args), config, stage="extract_embeddings"
    )

    print("[phase-1] Embedding extraction environment resolved successfully")
    print(f"hardware={env.hardware} device={env.device} split={args.split}")
    print(f"embed_dir={env.embed_dir}")
    print(f"config={config_path}")
    print(f"metadata={metadata_path}")
    print("[phase-1] Extraction loop is not implemented yet.")


if __name__ == "__main__":
    main()
