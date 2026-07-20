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
#   bash scripts/colab_run.sh [--gpu <type>] <stage> [extra args...]
#
# Examples:
#   bash scripts/colab_run.sh train_mgn --run-name colab-run --epochs 1
#   bash scripts/colab_run.sh train_sae --run-name colab-run --epochs 1
#   bash scripts/colab_run.sh --gpu L4 train_mgn --run-name colab-run --epochs 1
#   bash scripts/colab_run.sh --gpu "" train_mgn --run-name colab-run   # CPU runtime
#
# GPU selection (precedence: --gpu > COLAB_GPU env > preset default):
#   default is T4 (from configs/hardware/colab.yaml; free-tier compatible).
#   --gpu L4 / --gpu A100 (Pro+) override it; --gpu "" (or COLAB_GPU=) -> CPU.
#
# Env overrides:
#   COLAB_GPU   GPU type (alternative to --gpu).
#   COLAB_KEEP  1 = keep the VM alive after running (for interactive follow-up)
#
# Artifacts land under checkpoints/ embeddings/ runs/ (same layout as local),
# so a later local `analyze.py` can read them, or you can `--resume` on Colab.
set -euo pipefail

# Parse an optional leading --gpu [=]value before the stage.
GPU_FLAG=""
GPU_FLAG_UNSET=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_FLAG="${2:-}"; GPU_FLAG_UNSET=0; shift 2;;
    --gpu=*)
      GPU_FLAG="${1#--gpu=}"; GPU_FLAG_UNSET=0; shift;;
    --) shift; break;;
    *) break;;
  esac
done

STAGE="${1:?usage: colab_run.sh [--gpu <type>] <stage> [args...]}"
shift || true
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# GPU resolution: explicit --gpu > COLAB_GPU env > colab preset (T4).
# GPU_FLAG_UNSET == 1 means "--gpu" was NOT passed (distinct from "--gpu ''" = CPU).
if [[ "$GPU_FLAG_UNSET" == 1 ]]; then
  if [[ -n "${COLAB_GPU+x}" ]]; then
    GPU="$COLAB_GPU"   # may be empty -> CPU runtime
  else
    GPU="$(python3 -c "import sys; sys.path.insert(0,'.'); from src.config import load_hardware_config; print(load_hardware_config('.','colab').get('gpu','T4'))" 2>/dev/null || echo T4)"
  fi
else
  GPU="$GPU_FLAG"   # --gpu given (possibly empty -> CPU)
fi

# 1) Provision a fresh GPU VM (or CPU runtime when GPU is empty).
echo "==> colab new${GPU:+ --gpu $GPU}"
if [[ -n "$GPU" ]]; then
  colab new --gpu "$GPU"
else
  colab new
fi

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
