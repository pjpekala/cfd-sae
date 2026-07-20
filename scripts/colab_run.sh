#!/usr/bin/env bash
#
# Phase-4 Colab backend runner (powered by google-colab-cli).
#
# Provisions a GPU VM, uploads the repo, runs ONE pipeline stage there, and
# downloads the resulting artifacts back to this machine. The repo scripts are
# untouched -- colab-cli is just the transport; everything still runs via
# `uv run python scripts/<stage>.py --hardware colab`.
#
# NOTE: google-colab-cli's `upload`/`download` move a SINGLE file, and `exec`
# runs PYTHON code (not a shell). So we tar the repo + artifacts and shell out
# from Python via subprocess. That is the contract this wrapper honors.
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

KEEP="${COLAB_KEEP:-0}"

# Scratch files (kept off the repo).
TARBALL="$(mktemp /tmp/cfd-sae-repo-XXXXXX.tgz)"
RUNNER="$(mktemp /tmp/cfd-sae-runner-XXXXXX.py)"
ARTIFACTS="$(mktemp /tmp/cfd-sae-artifacts-XXXXXX.tgz)"
REMOTE_TARBALL=/content/cfd-sae.tgz
REMOTE_RUNNER=/content/cfd-sae/_colab_runner.py
REMOTE_ARTIFACTS=/content/cfd-sae/_artifacts.tgz
VM_DIR=/content/cfd-sae
cleanup() { rm -f "$TARBALL" "$RUNNER" "$ARTIFACTS"; }
trap cleanup EXIT

# 1) Provision a fresh GPU VM (or CPU runtime when GPU is empty).
echo "==> colab new${GPU:+ --gpu $GPU}"
if [[ -n "$GPU" ]]; then
  colab new --gpu "$GPU"
else
  colab new
fi

# 2) Package the repo (exclude heavy / ignored dirs) and upload + unpack.
echo "==> packaging repo (excluding .venv/data/checkpoints/embeddings/runs/.git)"
tar czf "$TARBALL" \
  --exclude=.venv --exclude=data --exclude=checkpoints \
  --exclude=embeddings --exclude=runs --exclude=.git \
  --exclude=.ruff_cache --exclude='__pycache__' --exclude='*.pyc' \
  -C "$REPO_ROOT" .

echo "==> colab upload $TARBALL $REMOTE_TARBALL"
colab upload "$TARBALL" "$REMOTE_TARBALL"

echo "==> colab exec: unpack repo into $VM_DIR"
# exec runs PYTHON. Unpack the tarball via subprocess.
printf 'import subprocess\nsubprocess.run(["tar","xzf","%s","-C","/content"],check=True)\n' "$REMOTE_TARBALL" | colab exec

# 3) Build a Python runner that shells out the stage (download data if needed).
#    Stages other than `analyze` need the dataset; fetch it on the VM.
NEEDS_DATA=1
[[ "$STAGE" == "analyze" ]] && NEEDS_DATA=0
RUN_CMD="cd $VM_DIR && uv sync"
if [[ "$NEEDS_DATA" == 1 ]]; then
  RUN_CMD="$RUN_CMD && uv run python scripts/download_data.py --data-dir data --skip-existing"
fi
RUN_CMD="$RUN_CMD && uv run python scripts/$STAGE.py --hardware colab ${EXTRA_ARGS[*]:-}"

# Write the runner (unquoted heredoc so RUN_CMD expands; it contains no ''').
cat > "$RUNNER" <<EOF
import subprocess
cmd = r'''${RUN_CMD}'''
print(">>", cmd)
subprocess.run(cmd, shell=True, check=True)
EOF

echo "==> colab upload $RUNNER $REMOTE_RUNNER"
colab upload "$RUNNER" "$REMOTE_RUNNER"
echo "==> colab exec --file $REMOTE_RUNNER"
colab exec --file "$REMOTE_RUNNER"

# 4) Pull artifacts back. Tar them on the VM, download the tarball, unpack.
echo "==> colab exec: tar artifacts on VM"
printf 'import subprocess\nsubprocess.run("tar czf %s -C %s checkpoints embeddings runs", shell=True, check=True)\n' \
  "$REMOTE_ARTIFACTS" "$VM_DIR" | colab exec

echo "==> colab download $REMOTE_ARTIFACTS $ARTIFACTS"
colab download "$REMOTE_ARTIFACTS" "$ARTIFACTS"

echo "==> unpacking artifacts into $REPO_ROOT"
mkdir -p "$REPO_ROOT"
tar xzf "$ARTIFACTS" -C "$REPO_ROOT"

if [[ "$KEEP" == "1" ]]; then
  echo "==> COLAB_KEEP=1: VM left running. Resume with 'colab exec ...' or stop with 'colab stop'."
else
  echo "==> colab stop"
  colab stop
fi

echo "==> done. Artifacts are under ./checkpoints ./embeddings ./runs"
