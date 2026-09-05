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
#   bash scripts/colab.sh [--dry-run] [--auth adc|oauth2] [--session <name>] [--repo-url <url>] <command> [args...]
#
# Commands:
#   new [--gpu T4|L4|A100|CPU]   Provision the VM (default GPU: T4)
#   drive                         Mount Google Drive (one-time browser consent)
#   sync                          Clone/pull the repo + uv sync on the VM
#   data                          Download cylinder-flow TFRecords to Drive (~16GB, once)
#   run <stage> [args...]         Run a pipeline stage on the VM
#   tensorboard <run-name> [--port 6006] [--poll] [--poll-interval 10] [--tb-only]  Start TB / start + poll
#   download <run-name> [--tb-only]  Pull a run's artifacts back (tb-only for fast TB poll)
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
AUTH=""

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

colab_cmd() {
  if [[ -n "$AUTH" ]]; then
    echo "colab --auth=$AUTH"
  else
    echo "colab"
  fi
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
    if [[ -n "$AUTH" ]]; then
      echo "[dry-run] colab --auth=$AUTH exec -s $SESSION --timeout $timeout <<'PY'"
    else
      echo "[dry-run] colab exec -s $SESSION --timeout $timeout <<'PY'"
    fi
    printf '%s\n' "$py"
    echo "PY"
  else
    if [[ -n "$AUTH" ]]; then
      printf '%s\n' "$py" | colab --auth="$AUTH" exec -s "$SESSION" --timeout "$timeout"
    else
      printf '%s\n' "$py" | colab exec -s "$SESSION" --timeout "$timeout"
    fi
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
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" new "${args[@]}"
  else
    run_local colab new "${args[@]}"
  fi
}

cmd_drive() {
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" drivemount -s "$SESSION"
  else
    run_local colab drivemount -s "$SESSION"
  fi
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

cmd_data() {
  local py
  printf -v py '%s\n' \
    "import os, subprocess" \
    "if not os.path.isdir('$VM_DIR'):" \
    "    raise SystemExit('repo not synced on VM; run colab.sh sync first')" \
    "if not os.path.isdir('/content/drive'):" \
    "    raise SystemExit('Drive not mounted; run colab.sh drive first')" \
    "cmd = 'cd $VM_DIR && uv run python scripts/download_data.py --data-dir $DRIVE_BASE/data --skip-existing 2>&1'" \
    "print('>>', cmd)" \
    "proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, bufsize=1)" \
    "assert proc.stdout is not None" \
    "for line in proc.stdout:" \
    "    print(line, end='')" \
    "proc.wait()" \
    "if proc.returncode != 0:" \
    "    print('[colab] data download exited with status %d' % proc.returncode)" \
    "    raise SystemExit(proc.returncode)"
  run_py "$py" 3600
}

cmd_run() {
  [[ $# -ge 1 ]] || { echo "usage: colab.sh run <stage> [stage args...]" >&2; exit 2; }
  local stage="$1"
  shift
  local py
  printf -v py '%s\n' \
    "import os, subprocess, sys" \
    "if not os.path.isdir('$VM_DIR'):" \
    "    raise SystemExit('repo not synced on VM; run colab.sh sync first')" \
    "if not os.path.isdir('/content/drive'):" \
    "    raise SystemExit('Drive not mounted; run colab.sh drive first')" \
    "if not os.path.exists('$DRIVE_BASE/data/train.tfrecord'):" \
    "    print('[colab] training data not on Drive; run bash scripts/colab.sh data first (~16GB)')" \
    "    sys.exit(2)" \
    "cmd = 'cd $VM_DIR && uv run python scripts/$stage.py --hardware colab $* 2>&1'" \
    "print('>>', cmd)" \
    "proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, bufsize=1)" \
    "assert proc.stdout is not None" \
    "for line in proc.stdout:" \
    "    print(line, end='')" \
    "proc.wait()" \
    "if proc.returncode != 0:" \
    "    print('[colab] stage exited with status %d' % proc.returncode)" \
    "    raise SystemExit(proc.returncode)"
  run_py "$py" "$TIMEOUT"
}

cmd_download() {
  [[ $# -ge 1 ]] || { echo "usage: colab.sh download <run-name> [--tb-only]" >&2; exit 2; }
  local run="$1"
  shift || true
  local tb_only=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tb-only) tb_only=1; shift ;;
      *) echo "Unknown arg for 'download': $1" >&2; exit 2;;
    esac
  done
  local remote="/content/cfd-sae-$run.tgz"
  if [[ "$tb_only" == 1 ]]; then
    remote="/content/cfd-sae-$run-tb.tgz"
  fi
  local tmp
  tmp="$(mktemp /tmp/cfd-sae-XXXXXX)"
  tmp="${tmp}.tgz"
  local py
  if [[ "$tb_only" == 1 ]]; then
    printf -v py '%s\n' \
      "import os, subprocess" \
      "base = '$DRIVE_BASE'" \
      "run = '$run'" \
      "tb_path = os.path.join(base, 'runs', run, 'tb')" \
      "if not os.path.isdir(tb_path):" \
      "    raise SystemExit(f'no tb logs on Drive for run {run!r}: {tb_path} (start training first)')" \
      "cmd = 'tar czf $remote -C %s runs/%s/tb' % (base, run)" \
      "proc = subprocess.run(cmd, shell=True)" \
      "if proc.returncode != 0 and not os.path.exists('$remote'):" \
      "    raise SystemExit(f'tar failed for {run!r} (code {proc.returncode})')" \
      "if proc.returncode != 0:" \
      "    print(f'[warn] tar exited {proc.returncode} but archive exists (file changed while writing?)')" \
      "print('tarred tb for run', run)"
  else
    printf -v py '%s\n' \
      "import os, subprocess" \
      "base = '$DRIVE_BASE'" \
      "run = '$run'" \
      "paths = [os.path.join(base, d, run) for d in ('checkpoints', 'embeddings', 'runs')]" \
      "missing = [p for p in paths if not os.path.isdir(p)]" \
      "if missing:" \
      "    raise SystemExit('no artifacts on Drive for run %r: %s' % (run, ', '.join(missing)))" \
      "cmd = 'tar czf $remote -C %s checkpoints/%s embeddings/%s runs/%s' % (base, run, run, run)" \
      "proc = subprocess.run(cmd, shell=True)" \
      "if proc.returncode != 0 and not os.path.exists('$remote'):" \
      "    raise SystemExit(f'tar failed for {run!r} (code {proc.returncode})')" \
      "if proc.returncode != 0:" \
      "    print(f'[warn] tar exited {proc.returncode} but archive exists')" \
      "print('tarred artifacts for run', run)"
  fi
  run_py "$py" 300
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" download -s "$SESSION" "$remote" "$tmp"
  else
    run_local colab download -s "$SESSION" "$remote" "$tmp"
  fi
  if [[ "$tb_only" == 1 ]]; then
    run_local mkdir -p "runs/$run/tb"
  else
    run_local mkdir -p checkpoints embeddings runs
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] tar xzf $tmp -C ."
  else
    tar xzf "$tmp" -C .
    rm -f "$tmp"
  fi
  if [[ "$tb_only" == 1 ]]; then
    echo "tb for run '$run' unpacked into ./runs/$run/tb"
  else
    echo "artifacts for run '$run' unpacked into ./checkpoints ./embeddings ./runs"
  fi
}

cmd_tensorboard() {
  local run="${1:-}"
  [[ -n "$run" ]] || { echo "usage: colab.sh tensorboard <run-name> [--port 6006] [--poll] [--poll-interval 10] [--tb-only]" >&2; exit 2; }
  shift || true
  local port="6006"
  local do_poll=0
  local poll_interval="10"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port) port="${2:?--port needs a value}"; shift 2;;
      --poll) do_poll=1; shift ;;
      --poll-interval) poll_interval="${2:?--poll-interval needs a value}"; shift 2;;
      --tb-only) shift ;;
      *) echo "Unknown arg for 'tensorboard': $1" >&2; exit 2;;
    esac
  done
  local logdir="$DRIVE_BASE/runs/$run/tb"
  local py
  printf -v py '%s\n' \
    "import os, subprocess, sys, time, socket" \
    "run = '$run'" \
    "logdir = '$logdir'" \
    "port = '$port'" \
    "if not os.path.isdir(logdir):" \
    "    print(f'[tb] no logdir yet: {logdir} (start training first; logs appear after first flush)')" \
    "    print(f'[tb] will still start server on :{port} polling {logdir}') " \
    "try:" \
    "    subprocess.run([\"bash\",\"-lc\",\"pkill -f 'tensorboard.*$run' || true\"], check=False)" \
    "except Exception:" \
    "    pass" \
    "cmd = f\"nohup tensorboard --logdir {logdir} --host 0.0.0.0 --port {port} --reload_interval 5 > /tmp/tb-{run}.log 2>&1 &\"" \
    "print(f'[tb] starting: tensorboard --logdir {logdir} --host 0.0.0.0 --port {port}')" \
    "proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)" \
    "print(proc.stdout, end='')" \
    "print(proc.stderr, end='')" \
    "time.sleep(2)" \
    "log = open(f'/tmp/tb-{run}.log').read()[-2000:] if os.path.exists(f'/tmp/tb-{run}.log') else ''" \
    "print(log)" \
    "print(f'[tb] logs on Drive: {logdir} (persistent, polled every 5s)')" \
    "print(f'[tb] VM TB at 0.0.0.0:{port} is INTERNAL to the VM (http://0.0.0.0:{port}/ is NOT clickable from your laptop)')" \
    "print(f'[tb] LIVE for CLI users: run LOCALLY: uv run tensorboard --logdir runs/{run}/tb --port {port}  # after: bash scripts/colab.sh download {run}  (or periodic pull for near-live)')" \
    "print(f'[tb] Colab notebook only: %load_ext tensorboard; %tensorboard --logdir {logdir}  # proxied to *.colab.googleusercontent.com')" \
    "print(f'[tb] delay: Drive sync ~5-10s; events flush every 20 steps')" \
    "sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)" \
    "try:" \
    "    sock.settimeout(1)" \
    "    sock.connect(('127.0.0.1', int(port)))" \
    "    print(f'[tb] VM listening on 0.0.0.0:{port} (internal, see above)')" \
    "except Exception as e:" \
    "    print(f'[tb] VM not yet listening on :{port}: {e}')" \
    "finally:" \
    "    sock.close()"
  run_py "$py" 60
  local auth_flag=""
  if [[ -n "$AUTH" ]]; then
    auth_flag=" --auth $AUTH"
  fi
  echo "[tb] LIVE: locally: uv run tensorboard --logdir runs/$run/tb --port $port  # then open http://localhost:$port  (polls Drive, 5-10s delay)"
  if [[ "$do_poll" == 1 ]]; then
    echo "[tb] --poll: starting local TensorBoard + Drive poll loop (interval ${poll_interval}s)"
    echo "[tb] polling: while true; do bash scripts/colab.sh${auth_flag} download $run --tb-only >/dev/null 2>&1; sleep $poll_interval; done &"
    if [[ "$DRY_RUN" == 1 ]]; then
      echo "[dry-run] uv run tensorboard --logdir runs/$run/tb --port $port &"
      echo "[dry-run] while true; do bash scripts/colab.sh${auth_flag} download $run --tb-only >/dev/null 2>&1; sleep $poll_interval; done &"
      echo "[dry-run] wait"
    else
      mkdir -p "runs/$run/tb"
      bash scripts/colab.sh${auth_flag} download "$run" --tb-only >/dev/null 2>&1 || true
      uv run tensorboard --logdir "runs/$run/tb" --port "$port" &
      local tb_pid=$!
      echo "[tb] local TensorBoard pid $tb_pid at http://localhost:$port"
      echo "[tb] polling Drive every ${poll_interval}s (Ctrl-C to stop)"
      trap "echo '[tb] stopping poll loop and TensorBoard'; kill $tb_pid 2>/dev/null || true; exit 0" INT TERM
      while true; do sleep "$poll_interval"; bash scripts/colab.sh${auth_flag} download "$run" --tb-only >/dev/null 2>&1 || true; done
      wait $tb_pid
    fi
  else
    echo "[tb] tip: for near-live, in another local terminal: while true; do bash scripts/colab.sh${auth_flag} download $run --tb-only >/dev/null 2>&1; sleep $poll_interval; done"
    echo "[tb] or: bash scripts/colab.sh${auth_flag} tensorboard $run --poll --poll-interval $poll_interval --port $port"
  fi
  echo "[tb] training: bash scripts/colab.sh${auth_flag} run train_mgn --run-name $run --epochs 25  (or train_sae)"
  echo "[tb] stop VM TB: colab${auth_flag} console -s $SESSION -> pkill -f tensorboard; cat /tmp/tb-$run.log  (not .lo)"
}

cmd_log() {
  local out="${1:-colab_run_log.md}"
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" log -s "$SESSION" -o "$out"
  else
    run_local colab log -s "$SESSION" -o "$out"
  fi
}

cmd_console() {
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" console -s "$SESSION"
  else
    run_local colab console -s "$SESSION"
  fi
}

cmd_status() {
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" status -s "$SESSION"
  else
    run_local colab status -s "$SESSION"
  fi
}

cmd_stop() {
  if [[ -n "$AUTH" ]]; then
    run_local colab --auth="$AUTH" stop -s "$SESSION"
  else
    run_local colab stop -s "$SESSION"
  fi
}

# ---- global flag parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auth)
      val="${2:-}"
      if [[ "$val" == "adc" || "$val" == "oauth2" ]]; then
        if [[ "$val" == "adc" ]]; then AUTH="adc"; else AUTH=""; fi
        shift 2
      elif [[ -z "$val" || "$val" == -* ]]; then
        AUTH="adc"
        shift
      else
        echo "Unknown --auth value: $val (use adc or oauth2)" >&2; exit 2
      fi
      ;;
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
  data) cmd_data ;;
  run) cmd_run "$@" ;;
  tensorboard) cmd_tensorboard "$@" ;;
  download) cmd_download "$@" ;;
  log) cmd_log "$@" ;;
  console) cmd_console ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  "") usage ;;
  *) echo "Unknown command: $CMD" >&2; usage >&2; exit 2 ;;
esac
