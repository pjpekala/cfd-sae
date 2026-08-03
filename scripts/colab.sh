#!/usr/bin/env bash
#
# Colab session manager (git + Drive model).
#
# Manages ONE persistent, Drive-mounted Colab VM and runs pipeline stages
# against it. Code reaches the VM via git (no tarball upload); artifacts live
# on Google Drive (source of truth) and can be pulled back locally on demand.
#
# No environment variables. All knobs are flags or hardcoded defaults.
#
# Usage:
#   bash scripts/colab.sh [--dry-run] [--session <name>] [--repo-url <url>] <command> [args...]
#
# Commands:
#   new [--gpu T4|L4|A100|CPU]   Provision the VM (default GPU: T4)
#   drive                         Mount Google Drive (one-time browser consent)
#   sync                          Clone/pull the repo + uv sync on the VM
#   run <stage> [args...]         Run a pipeline stage on the VM
#   download <run-name>           Pull a run's artifacts back to ./checkpoints ./embeddings ./runs
#   log [output]                  Export a replayable log of the session
#   console                       Interactive debug shell on the VM
#   status                        Show session status
#   stop                          Release the VM
#
# --dry-run prints every command that would run, without executing anything.
#
# Manual workflow (one-time):
#   uv tool install google-colab-cli
#   colab auth
#   bash scripts/colab.sh new
#   bash scripts/colab.sh drive        # approve in browser, press Enter
#   bash scripts/colab.sh sync
#   bash scripts/colab.sh run train_mgn --run-name myrun --epochs 25
#   bash scripts/colab.sh download myrun   # optional local copy
#   bash scripts/colab.sh stop
set -euo pipefail

# ---- defaults (override with flags, not env vars) ----
SESSION="cfd"
VM_DIR="/content/cfd-sae"
DRIVE_BASE="/content/drive/MyDrive/cfd-sae"
TIMEOUT=3600
GPU="T4"

DRY_RUN=0
REPO_URL=""

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

# Run a local command, or print it in dry-run mode.
run_local() {
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

# Send python code to the VM via colab exec, or print it in dry-run mode.
# Args: <python source> [timeout]
run_py() {
  local py="$1"
  local timeout="${2:-30}"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] colab exec -s $SESSION --timeout $timeout <<'PY'"
    printf '%s\n' "$py"
    echo "PY"
  else
    printf '%s\n' "$py" | colab exec -s "$SESSION" --timeout "$timeout"
  fi
}

# Resolve the repo URL from the local git remote 'origin', converting
# ssh/git forms to https so the VM can clone anonymously.
repo_url_from_git() {
  local url
  url="$(git remote get-url origin 2>/dev/null || true)"
  [[ -z "$url" ]] && return 1
  if [[ "$url" == git@*:* ]]; then
    url="https://${url#git@}"
    url="${url/:/\/}"
  fi
  url="${url/git:\/\//https:\/\/}"
  printf '%s' "$url"
}

cmd_new() {
  local gpu=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu) gpu="${2:-}"; shift 2;;
      *) echo "Unknown arg for 'new': $1" >&2; usage >&2; exit 2;;
    esac
  done
  [[ -n "$gpu" ]] || gpu="$GPU"
  [[ "$gpu" != "CPU" ]] || gpu=""
  local args=(-s "$SESSION")
  [[ -n "$gpu" ]] && args+=(--gpu "$gpu")
  run_local colab new "${args[@]}"
}

cmd_drive() {
  run_local colab drivemount -s "$SESSION"
}

cmd_sync() {
  local repo="${REPO_URL:-$(repo_url_from_git || true)}"
  if [[ -z "$repo" ]]; then
    echo "error: no git remote 'origin'. Set one (git remote add origin <url>) or pass --repo-url <url>." >&2
    exit 2
  fi
  if [[ "$repo" == *"'"* || "$repo" == *[[:space:]]* ]]; then
    echo "error: repo URL must not contain single quotes or whitespace: $repo" >&2
    exit 2
  fi
  local py
  printf -v py '%s\n' \
    "import os, shutil, subprocess" \
    "vm = '$VM_DIR'" \
    "repo = '$repo'" \
    "if not os.path.isdir(vm):" \
    "    subprocess.run(['git', 'clone', '--depth', '1', repo, vm], check=True)" \
    "else:" \
    "    subprocess.run(['git', '-C', vm, 'pull', '--ff-only'], check=True)" \
    "if shutil.which('uv') is None:" \
    "    subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)" \
    "    os.environ['PATH'] = os.path.expanduser('~/.local/bin') + os.pathsep + os.environ.get('PATH', '')" \
    "subprocess.run(['bash', '-lc', 'cd $VM_DIR && uv sync'], check=True)"
  run_py "$py" 600
}

cmd_run() {
  [[ $# -ge 1 ]] || { echo "usage: colab.sh run <stage> [stage args...]" >&2; exit 2; }
  local stage="$1"
  shift
  local py
  printf -v py '%s\n' \
    "import os, subprocess" \
    "if not os.path.isdir('$VM_DIR'):" \
    "    raise SystemExit('repo not synced on VM; run colab.sh sync first')" \
    "if not os.path.isdir('/content/drive'):" \
    "    raise SystemExit('Drive not mounted; run colab.sh drive first')" \
    "cmd = 'cd $VM_DIR && uv run python scripts/$stage.py --hardware colab $*'" \
    "print('>>', cmd)" \
    "subprocess.run(cmd, shell=True, check=True)"
  run_py "$py" "$TIMEOUT"
}

cmd_download() {
  [[ $# -ge 1 ]] || { echo "usage: colab.sh download <run-name>" >&2; exit 2; }
  local run="$1"
  local remote="/content/cfd-sae-$run.tgz"
  local tmp
  tmp="$(mktemp /tmp/cfd-sae-XXXXXX.tgz)"
  local py
  printf -v py '%s\n' \
    "import os, subprocess" \
    "base = '$DRIVE_BASE'" \
    "run = '$run'" \
    "paths = [os.path.join(base, d, run) for d in ('checkpoints', 'embeddings', 'runs')]" \
    "missing = [p for p in paths if not os.path.isdir(p)]" \
    "if missing:" \
    "    raise SystemExit('no artifacts on Drive for run %r: %s' % (run, ', '.join(missing)))" \
    "cmd = 'tar czf $remote -C %s checkpoints/%s embeddings/%s runs/%s' % (base, run, run, run)" \
    "subprocess.run(cmd, shell=True, check=True)" \
    "print('tarred artifacts for run', run)"
  run_py "$py" 300
  run_local colab download -s "$SESSION" "$remote" "$tmp"
  run_local mkdir -p checkpoints embeddings runs
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] tar xzf $tmp -C ."
  else
    tar xzf "$tmp" -C .
    rm -f "$tmp"
  fi
  echo "artifacts for run '$run' unpacked into ./checkpoints ./embeddings ./runs"
}

cmd_log() {
  local out="${1:-colab_run_log.md}"
  run_local colab log -s "$SESSION" -o "$out"
}

cmd_console() {
  run_local colab console -s "$SESSION"
}

cmd_status() {
  run_local colab status -s "$SESSION"
}

cmd_stop() {
  run_local colab stop -s "$SESSION"
}

# ---- global flag parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --session) SESSION="${2:?--session needs a value}"; shift 2 ;;
    --repo-url) REPO_URL="${2:?--repo-url needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done

CMD="${1:-}"
shift 2>/dev/null || true

case "$CMD" in
  new) cmd_new "$@" ;;
  drive) cmd_drive ;;
  sync) cmd_sync ;;
  run) cmd_run "$@" ;;
  download) cmd_download "$@" ;;
  log) cmd_log "$@" ;;
  console) cmd_console ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  "") usage ;;
  *) echo "Unknown command: $CMD" >&2; usage >&2; exit 2 ;;
esac
