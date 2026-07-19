#!/usr/bin/env python3
"""Portable data download script for cylinder flow TFRecords."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download cylinder flow TFRecords.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        target = data_dir / f"{split}.tfrecord"
        if target.exists() and args.skip_existing:
            print(f"Skipping existing file: {target}")
            continue

        url = f"{BASE_URL}/{split}.tfrecord"
        print(f"Downloading {url} -> {target}")
        urlretrieve(url, target)

    print(f"Done. Files in {data_dir}")


if __name__ == "__main__":
    main()
