"""Shared CLI argument helpers for pipeline scripts."""

from __future__ import annotations

import argparse

from src.env import HARDWARE_CHOICES


def add_common_args(parser: argparse.ArgumentParser, include_resume: bool = True) -> None:
    parser.add_argument(
        "--hardware",
        default="auto",
        choices=HARDWARE_CHOICES,
        help="Hardware preset to use.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run identifier for isolated artifacts.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base dir for data/checkpoints/embeddings (default: repo root, or Drive on Colab).",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    if include_resume:
        parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
