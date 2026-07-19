#!/usr/bin/env bash
#
# Phase-4 Colab backend runner (powered by google-colab-cli).
#
# Provisions a GPU VM, uploads the repo, runs ONE pipeline stage there, and
# downloads the resulting artifacts back to this machine. The repo scripts are
# untouched -- colab-cli is just the transport; everything still runs via
# `uv run python scripts/<stage>.py --hardware colab`.
#
# Usage:
#   bash scripts/colab_run.sh <stage> [extra args...]
#
# Examples:
#   bash scripts/colab_run.sh train_mgn --run-name colab-run --epochs 1
#   bash scripts/colab_run.sh extract_embeddings --run-name colab-run --split test
#   bash scripts/colab_run.sh train_sae --run-name colab-run --epochs 1
#   bash scripts/colab_run.sh analyze --run-name colab-run --split test
#
# Env overrides:
#   COLAB_GPU   GPU type passed to `colab new` (default: A100)
#   COLAB_KEEP  1 = keep the VM alive after running (for interactive follow-up)
#
# Artifacts land under checkpoints/ embeddings/ runs/ (same layout as local),
# so a later local `analyze.py` can read them, or you can `--resume` on Colab.
set -euo pipefail

STAGE="${1:?usage: colab_run.sh <stage> [args...]}"
shift || true
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU="${COLAB_GPU:-A100}"
KEEP="${COLAB_KEEP:-0}"

# 1) Provision a fresh GPU VM (auto-uploads nothing yet -- explicit upload below).
echo "==> colab new --gpu $GPU"
colab new --gpu "$GPU"

# 2) Upload the repo into the VM (colab-cli uploads relative to cwd).
echo "==> colab upload . /content/cfd-sae"
colab upload . /content/cfd-sae

# 3) Run the requested stage on the VM.
echo "==> colab exec: train stage=$STAGE"
colab exec "cd /content/cfd-sae && uv sync && uv run python scripts/$STAGE.py --hardware colab ${EXTRA_ARGS[*]:-}"

# 4) Pull artifacts back. checkpoints/ embeddings/ runs/ mirror the local layout.
echo "==> colab download checkpoints embeddings runs ./"
colab download checkpoints embeddings runs ./

if [[ "$KEEP" == "1" ]]; then
  echo "==> COLAB_KEEP=1: VM left running. Resume with 'colab exec ...' or stop with 'colab stop'."
else
  echo "==> colab stop"
  colab stop
fi

echo "==> done. Artifacts are under ./checkpoints ./embeddings ./runs"
