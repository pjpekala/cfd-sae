# CFD-SAE: How to Use This Repo

A practical, step-by-step guide to running the **CFD-SAE** pipeline: training a
MeshGraphNet (MGN) on cylinder-flow CFD data, extracting node embeddings,
training a Sparse Autoencoder (SAE), and running interpretability analysis.

This pipeline implements the approach from
[arXiv:2507.16069](https://arxiv.org/abs/2507.16069) (Hu & Liu, IJCAI 2025
XAI workshop). It is designed to be **colab-first, resumable, and reproducible**.

---

## Table of Contents

1. [What Is CFD-SAE?](#1-what-is-cfd-sae)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Quick Start (Local)](#4-quick-start-local)
5. [Pipeline Stages](#5-pipeline-stages)
6. [Hardware Presets](#6-hardware-presets)
7. [Run Management](#7-run-management)
8. [Colab Usage](#8-colab-usage)
9. [Interactive Analysis (Notebooks)](#9-interactive-analysis-notebooks)
10. [Configuration](#10-configuration)
11. [Artifact Layout](#11-artifact-layout)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What Is CFD-SAE?

CFD-SAE is a four-stage pipeline that trains a neural model on fluid dynamics
data and then interprets what the model has learned using a sparse autoencoder:

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────┐
│  train_mgn  │ ──▶ │ extract_embeddings│ ──▶ │  train_sae  │ ──▶ │ analyze  │
│             │     │                  │     │             │     │          │
│ Train a     │     │ Run the trained  │     │ Train an    │     │ Compute  │
│ MeshGraphNet│     │ MGN over a data  │     │ SAE on the  │     │ Top-K    │
│ on cylinder │     │ split and save   │     │ frozen node │     │ salient  │
│ flow data   │     │ pre-decoder      │     │ embeddings  │     │ latents  │
│             │     │ embeddings h_i   │     │             │     │          │
└─────────────┘     └──────────────────┘     └─────────────┘     └──────────┘
```

**Stage 1 — `train_mgn.py`**: Trains a MeshGraphNet to predict the next frame of
velocity and pressure on a 2D cylinder-flow mesh. The model uses 9 message-passing
steps with hidden dimension 128 (paper specification).

**Stage 2 — `extract_embeddings.py`**: Runs the trained MGN in inference mode and
saves the **pre-decoder node embeddings** `h_i` (the frozen representation the
SAE trains on) as chunked `.npy` files.

**Stage 3 — `train_sae.py`**: Trains a sparse autoencoder (Linear→ReLU encoder,
Linear decoder with unit-norm dictionary atoms) on the extracted embeddings.
Loss = reconstruction MSE + `lambda_l1` * L1 sparsity penalty. Includes
early-stopping on a held-out validation set.

**Stage 4 — `analyze.py`**: Encodes all embeddings through the trained SAE and
computes interpretability diagnostics: reconstruction MSE, mean L1 of codes,
fraction of inactive latents, and Top-K salient latents ranked by three
Table-1 scores (Variance, MeanAbs, Entropy).

All four stages chain on a shared `--run-name` so artifacts line up across stages.

---

## 2. Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.12 (see `.python-version`) |
| **uv** | [Install uv](https://docs.astral.sh/uv/) — used for all dependency and environment management |
| **Data** | Cylinder-flow TFRecords (downloaded automatically, ~500 MB total) |
| **GPU (optional)** | CUDA GPU for faster training. CPU/MPS also work but are slower. |

---

## 3. Installation

This project uses `uv` for everything. Do **not** use system `pip`.

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/YOUR_USERNAME/cfd-sae
cd cfd-sae

# Install all dependencies (creates .venv automatically)
uv sync
```

This installs:
- **Runtime deps**: torch, torch-geometric, numpy, matplotlib, scipy, tfrecord, tqdm, pyyaml
- **Dev deps**: ruff (linter), google-colab-cli (Colab backend), ipywidgets (interactive notebooks)

### Verify the installation

```bash
# Check that the CLI wiring works
uv run python scripts/train_mgn.py --help

# You should see:
# usage: train_mgn.py [-h] [--hardware {auto,colab,desktop,macbook}]
#                     [--run-name RUN_NAME] [--seed SEED] [--resume]
#                     [--epochs EPOCHS] [--max-steps MAX_STEPS]
#                     [--save-every SAVE_EVERY]
```

---

## 4. Quick Start (Local)

This example runs the full pipeline end-to-end on a single machine with small
epoch/step limits so it completes quickly. Replace `macbook` with `desktop` or
`colab` as appropriate for your machine.

### Step 1: Download the data (once)

```bash
uv run python scripts/download_data.py --data-dir data --skip-existing
```

This downloads three TFRecord files:
- `data/train.tfrecord` (~250 MB)
- `data/valid.tfrecord` (~50 MB)
- `data/test.tfrecord` (~50 MB)

The `--skip-existing` flag skips files that already exist, making this safe to
re-run.

**Alternative (shell):**
```bash
bash scripts/download_data.sh ./data
```

### Step 2: Train the MGN (smoke test)

```bash
uv run python scripts/train_mgn.py \
    --hardware macbook \
    --run-name smoke-run \
    --epochs 1
```

**Flags explained:**
- `--hardware macbook` — Use the macbook preset (CPU/MPS, small batch sizes).
  Use `auto` to auto-detect, or `desktop` for a PC with a GPU.
- `--run-name smoke-run` — All artifacts for this run go under
  `checkpoints/smoke-run/`, `embeddings/smoke-run/`, `runs/smoke-run/`.
- `--epochs 1` — Train for 1 epoch (use 50+ for real training on desktop).

**Expected output:**
```
[train] MGN params: 423,810
hardware=macbook device=mps run_name=smoke-run
epoch=1 step=20 loss=0.012345
epoch=1 step=40 loss=0.009876
...
[train] done. steps=120 best_loss=0.004567 nan=False
  checkpoint=checkpoints/smoke-run/epoch_0001.pt
```

### Step 3: Extract embeddings

```bash
uv run python scripts/extract_embeddings.py \
    --hardware macbook \
    --run-name smoke-run \
    --split test
```

**Flags explained:**
- `--split test` — Extract embeddings from the test split (default; this is what
  the paper uses for SAE training). Use `train` or `valid` if needed.

**Expected output:**
```
[extract] split=test files=1200 node_vectors=2307600
  out_dir=embeddings/smoke-run/test
```

This creates files like `embeddings/smoke-run/test/ex00000_fr0000.npy`, each
containing a `[N, 128]` float32 array of node embeddings for one (example, frame).

### Step 4: Train the SAE

```bash
uv run python scripts/train_sae.py \
    --hardware macbook \
    --run-name smoke-run \
    --epochs 1 \
    --max-steps 100
```

**Flags explained:**
- `--epochs 1` — Train for 1 epoch (use 50+ for real training).
- `--max-steps 100` — Cap at 100 optimizer steps (smoke test only).
- `--val-frac 0.1` — Hold out 10% of embedding files for early-stopping (default).
- `--patience 5` — Stop after 5 epochs without val-loss improvement (default).
- `--no-val` — Disable early-stopping; run fixed epochs.

**Expected output:**
```
[sae] embedding files: 1200 (split=test) train=1080 val=120
[sae] computed embedding norm over 1080 train files -> embedding_stats.npz
epoch=1 step=20 loss=0.023456
epoch=1 step=40 loss=0.019876
...
[sae] done. steps=100 best_train_loss=0.012345 best_val=0.015678 early_stop=False
```

### Step 5: Run analysis

```bash
uv run python scripts/analyze.py \
    --hardware macbook \
    --run-name smoke-run \
    --split test \
    --top-k 20
```

**Flags explained:**
- `--top-k 20` — Report the top 20 most salient latents per score (default 20).
- `--entropy-bins 50` — Number of histogram bins for the Entropy score (default 50).

**Expected output:**
```
[analyze] split=test samples=2307600 hidden=1024
  recon_mse=0.015678 mean_l1=0.234567 frac_inactive=0.789
  Top-20 by Variance: [42, 17, 88, 103, ...] ...
  Top-20 by MeanAbs:  [42, 17, 256, 103, ...] ...
  results=runs/smoke-run/analysis.json
```

The full analysis is saved to `runs/smoke-run/analysis.json`.

---

## 5. Pipeline Stages

### `train_mgn.py` — Train MeshGraphNet

Trains the MGN on next-frame prediction (velocity + pressure) over the train
split. Uses periodic + best checkpointing with `--resume` support.

```bash
uv run python scripts/train_mgn.py \
    --hardware desktop \
    --run-name my-mgn \
    --epochs 50 \
    --save-every 200 \
    --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--hardware` | `auto` | Hardware preset (see [§6](#6-hardware-presets)) |
| `--run-name` | auto-generated | Run identifier for artifact isolation |
| `--seed` | `17` | Random seed for reproducibility |
| `--resume` | off | Resume from latest checkpoint (requires `--run-name`) |
| `--epochs` | from preset | Number of training epochs |
| `--max-steps` | off | Cap optimizer steps (smoke tests) |
| `--save-every` | `200` | Checkpoint every N global steps |

**Resume example:**
```bash
# If the run was interrupted, continue from the last checkpoint:
uv run python scripts/train_mgn.py \
    --hardware desktop \
    --run-name my-mgn \
    --resume \
    --epochs 50
```

**Key behavior:** `--resume` restores the model state, optimizer state, and
`global_step` from the latest checkpoint in `checkpoints/<run-name>/`. It also
validates that the resolved config matches the saved snapshot (see
[§7.2](#72-resume-contract)).

### `extract_embeddings.py` — Extract Node Embeddings

Loads the trained MGN (`best.pt`) and runs inference over a data split, saving
pre-decoder node embeddings `h_i` as chunked `.npy` files.

```bash
uv run python scripts/extract_embeddings.py \
    --hardware desktop \
    --run-name my-mgn \
    --split test \
    --max-examples 10
```

| Flag | Default | Description |
|---|---|---|
| `--split` | `test` | Which data split to extract (`train`, `valid`, or `test`) |
| `--max-examples` | off | Cap number of examples (smoke tests) |
| `--mgn-run` | same as `--run-name` | Use a different run's MGN checkpoint |

**Using a different MGN run:**
```bash
# Extract embeddings using MGN from run "mgn-exp-1" but save under "emb-exp-1"
uv run python scripts/extract_embeddings.py \
    --hardware desktop \
    --run-name emb-exp-1 \
    --mgn-run mgn-exp-1 \
    --split test
```

### `train_sae.py` — Train Sparse Autoencoder

Trains an SAE on the extracted embeddings with decoder unit-L2-norm constraint
and early-stopping on held-out validation.

```bash
uv run python scripts/train_sae.py \
    --hardware desktop \
    --run-name my-sae \
    --epochs 50 \
    --val-frac 0.1 \
    --patience 5 \
    --min-epochs 1
```

| Flag | Default | Description |
|---|---|---|
| `--split` | `test` | Which extracted split to train on |
| `--val-frac` | `0.1` | Fraction of embedding files held out for validation |
| `--patience` | `5` | Stop after N epochs without val-loss improvement |
| `--min-epochs` | `1` | Minimum epochs before early-stopping can trigger |
| `--no-val` | off | Disable early-stopping; run fixed `--epochs` |
| `--mgn-run` | same as `--run-name` | Use embeddings from a different run |
| `--max-steps` | off | Cap optimizer steps (smoke tests) |

**Disable early-stopping (fixed epochs):**
```bash
uv run python scripts/train_sae.py \
    --hardware desktop \
    --run-name my-sae \
    --epochs 50 \
    --no-val
```

**Resume SAE training:**
```bash
uv run python scripts/train_sae.py \
    --hardware desktop \
    --run-name my-sae \
    --resume \
    --epochs 50
```

**Key behavior:** The SAE decoder rows (dictionary atoms) are re-normalized to
unit L2 norm after every optimizer step. Embedding z-score normalization stats
are computed over the **train subset only** and persisted to
`embeddings/<run>/embedding_stats.npz` so analysis and validation reuse the
identical transform.

### `analyze.py` — Interpretability Analysis

Loads the trained SAE + embeddings and computes diagnostics.

```bash
uv run python scripts/analyze.py \
    --hardware desktop \
    --run-name my-sae \
    --split test \
    --top-k 20 \
    --entropy-bins 50
```

| Flag | Default | Description |
|---|---|---|
| `--split` | `test` | Which split to analyze |
| `--top-k` | `20` | Number of top latents to report per score |
| `--entropy-bins` | `50` | Histogram bins for the Entropy score |
| `--mgn-run` | same as `--run-name` | Use embeddings/SAE from a different run |

**Output file:** `runs/<run-name>/analysis.json` contains:
- `recon_mse` — mean reconstruction MSE
- `mean_l1_codes` — mean L1 norm of latent codes (sparsity proxy)
- `frac_inactive_codes` — fraction of near-zero activations
- `top_k_latents` — top-K latent indices per score
- `score_variance_topk`, `score_meanabs_topk`, `score_entropy_topk` — score values

---

## 6. Hardware Presets

Hardware presets live in `configs/hardware/` as YAML files. Each preset sets
batch sizes, worker counts, and model hyperparameters appropriate for the
target machine.

| Preset | File | GPU | MGN epochs | SAE batch | Use case |
|---|---|---|---|---|---|
| `auto` | (resolved at runtime) | auto-detect | — | — | Auto-selects based on environment |
| `colab` | `configs/hardware/colab.yaml` | T4 (default) | 25 | 128 | Google Colab (free tier) |
| `desktop` | `configs/hardware/desktop.yaml` | CUDA | 50 | 256 | PC with dedicated GPU |
| `macbook` | `configs/hardware/macbook.yaml` | MPS/CPU | 5 | 64 | Apple Silicon / laptop |

All presets use MGN `hidden_dim: 128` and `message_passing_steps: 9` (paper spec).

### Auto-detection

When you pass `--hardware auto`, the pipeline resolves to:
- `colab` if running inside Google Colab
- `macbook` if on macOS (Darwin)
- `desktop` otherwise

### Device resolution

The device is auto-detected at runtime:
- `cuda` if a CUDA GPU is available
- `mps` if Apple Silicon GPU is available (macbook preset)
- `cpu` as fallback

You can see the resolved device in the script output:
```
hardware=desktop device=cuda run_name=my-run
```

---

## 7. Run Management

### 7.1 Run Names and Artifact Isolation

Every run is isolated under a run name. All artifacts for a run live under:

```
checkpoints/<run_name>/    # MGN + SAE checkpoints
embeddings/<run_name>/     # Extracted node embeddings
runs/<run_name>/           # Metadata, config snapshot, analysis results
```

**Auto-generated run names** follow the format:
```
<stage>-<YYYYMMDD-HHMMSS>-<hardware>-s<seed>
```
For example: `mgn-20260728-143022-desktop-s17`

**Explicit run names** let you chain stages together:
```bash
# All stages use the same run name so artifacts line up
uv run python scripts/train_mgn.py         --run-name my-exp --epochs 50
uv run python scripts/extract_embeddings.py --run-name my-exp --split test
uv run python scripts/train_sae.py         --run-name my-exp --epochs 50
uv run python scripts/analyze.py           --run-name my-exp --split test
```

### 7.2 Resume Contract

`--resume` continues `train_mgn` and `train_sae` from the latest checkpoint.

**Requirements:**
- `--resume` **requires** an explicit `--run-name`.
  Auto-generated names cannot be resumed because they are unique per run.
- The resolved config must be compatible with the saved snapshot. Critical keys
  (`mgn`, `sae`, `batch_size`) are compared.

**Example:**
```bash
# Start a run
uv run python scripts/train_mgn.py --hardware desktop --run-name long-run --epochs 50

# ... (interrupted by a crash or Colab disconnect) ...

# Resume — continues from the last checkpoint
uv run python scripts/train_mgn.py --hardware desktop --run-name long-run --resume --epochs 50
```

**Config mismatch error:**
```
ValueError: Resume config mismatch for keys: mgn. Use a new --run-name for incompatible changes.
```
This means the current config differs from the saved snapshot for a critical key.
Fix: use a new `--run-name` for the changed config, or restore the original config.

### 7.3 Run Metadata

Each run produces metadata files in `runs/<run-name>/`:

- **`resolved_config.yaml`** — The full resolved hardware config (preset + CLI overrides).
- **`run_metadata.json`** — Timestamp, git commit, env info, args, and config snapshot.
- **`analysis.json`** (analyze stage only) — Full analysis results.

Example `run_metadata.json`:
```json
{
  "created_at_utc": "2026-07-28T14:30:22.123456+00:00",
  "stage": "train_mgn",
  "git": {"commit": "a1b2c3d", "dirty": false},
  "env": {
    "hardware": "desktop",
    "device": "cuda",
    "run_name": "my-exp",
    "ckpt_dir": "/path/to/checkpoints/my-exp",
    "embed_dir": "/path/to/embeddings/my-exp",
    "run_dir": "/path/to/runs/my-exp"
  },
  "args": {"hardware": "desktop", "run_name": "my-exp", "epochs": 50, "seed": 17},
  "config": { ... }
}
```

---

## 8. Colab Usage

The recommended way to run on Colab is via `google-colab-cli`, which manages a
**persistent, Drive-mounted GPU VM**. Code is synced via git and artifacts live
on Google Drive, so stages chain naturally and `--resume` works across sessions.
No notebooks required.

### Install the CLI (one time)

```bash
uv tool install google-colab-cli
```

See "Authentication & Drive account" below before provisioning a VM.

### Authentication & Drive account

The Drive mounted by `colab.sh drive` / `colab drivemount` belongs to the
**Google account the CLI is authenticated as** — not the VM's account. Check it
before downloading data (so ~16 GB lands in the right account's quota):

```bash
colab whoami          # hidden debug cmd: active email, scopes, expiry
```

`colab auth` is unrelated to CLI authentication — it injects *VM-side* GCP
credentials (BigQuery/GCS). Don't use it to "log in" the CLI.

The CLI supports two auth strategies; the flag goes **before** the subcommand.

#### oauth2 (default)

One-time browser consent flow; token cached at `~/.config/colab-cli/token.json`.
To switch accounts:

```bash
rm ~/.config/colab-cli/token.json     # clear cache
colab whoami                          # re-opens browser; pick the new account
```

If a VM was already provisioned under the old account, stop it before switching
(it stays reachable only under that account):

```bash
colab stop -s cfd
```

#### adc (Application Default Credentials)

Recommended when the target Drive lives in a specific account (e.g., a student
account with more space). Authenticate gcloud ADC as that account with **all
four** scopes the Colab backend requires — a plain
`gcloud auth application-default login` is missing `colaboratory`, which makes
`colab new` 403 and unassigns the fresh VM:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

`colab.sh` does **not** pass `--auth` (it uses the default oauth2), so with ADC
you drive the VM with raw `colab --auth=adc ...` commands — see "Using ADC (raw
commands)" below.

### Provision + mount Drive (one time per VM)

```bash
bash scripts/colab.sh new          # T4 GPU (free tier); --gpu L4/A100/CPU to change
bash scripts/colab.sh drive        # approve in browser, press Enter
```

### Sync code + deps

```bash
bash scripts/colab.sh sync         # clones/pulls this repo; re-run after each git push
```

The repo URL is taken from your local `git remote get-url origin` (ssh/git URLs
are converted to https for the VM's anonymous clone). Override with `--repo-url`
if needed.

### Run pipeline stages

```bash
bash scripts/colab.sh run train_mgn          --run-name colab-run --epochs 25
bash scripts/colab.sh run extract_embeddings --run-name colab-run --split test
bash scripts/colab.sh run train_sae          --run-name colab-run --epochs 25
bash scripts/colab.sh run analyze            --run-name colab-run --split test
```

`run` guards on the repo being synced and Drive being mounted. Artifacts are
written to `/content/drive/MyDrive/cfd-sae/` on the VM's mounted Drive, so they
survive VM teardown and stages can `--resume` on a later VM.

### Pull artifacts back (optional — Drive is the source of truth)

```bash
bash scripts/colab.sh download colab-run
```

Downloads that run's `checkpoints/`, `embeddings/`, and `runs/` from Drive into
your local repo.

### Release the VM

```bash
bash scripts/colab.sh stop
```

### Other commands

```bash
bash scripts/colab.sh log             # tail the VM's sync/run logs
bash scripts/colab.sh console         # interactive shell on the VM (use `exit` to leave)
bash scripts/colab.sh status          # show VM/Drive/session state
bash scripts/colab.sh drive           # (re)mount Google Drive
```

### GPU selection

By default, `colab.sh new` requests a **T4** GPU (free-tier compatible).
Override at VM creation time:

```bash
bash scripts/colab.sh new --gpu L4      # Colab Pro+
bash scripts/colab.sh new --gpu A100    # Colab Pro+ (higher tiers)
bash scripts/colab.sh new --gpu CPU     # CPU-only runtime
```

### Using ADC (raw commands)

If you authenticate via ADC (`--auth=adc`), bypass `colab.sh` (it uses the
default oauth2) and drive the VM directly. Every command needs `--auth=adc`
before the subcommand, and you must use it consistently — the keep-alive daemon
inherits the auth used at `colab new`, and mixing strategies is unreliable.

```bash
# Verify identity + scopes
colab --auth=adc whoami

# Provision + mount Drive (this mounts the ADC account's Drive)
colab --auth=adc new -s cfd --gpu T4
colab --auth=adc drivemount -s cfd
colab --auth=adc ls -s cfd /content/drive/MyDrive     # confirm the account's Drive

# Sync repo + deps
colab --auth=adc exec -s cfd --timeout 600 <<'PY'
import os, shutil, subprocess
vm = '/content/cfd-sae'
repo = 'https://github.com/pjpekala/cfd-sae.git'
if not os.path.isdir(vm):
    subprocess.run(['git', 'clone', '--depth', '1', repo, vm], check=True)
else:
    subprocess.run(['git', '-C', vm, 'pull', '--ff-only'], check=True)
if shutil.which('uv') is None:
    subprocess.run('curl -LsSf https://astral.sh/uv/install.sh | sh', shell=True, check=True)
    os.environ['PATH'] = os.path.expanduser('~/.local/bin') + os.pathsep + os.environ.get('PATH', '')
subprocess.run(['bash', '-lc', 'cd /content/cfd-sae && uv sync'], check=True)
PY

# Download data into the mounted Drive (~16 GB)
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/download_data.py --data-dir /content/drive/MyDrive/cfd-sae/data --skip-existing", shell=True, check=True)\n' \
  | colab --auth=adc exec -s cfd --timeout 3600

# Run a pipeline stage (same pattern for extract_embeddings / train_sae / analyze)
printf 'import subprocess; subprocess.run("cd /content/cfd-sae && uv run python scripts/train_mgn.py --hardware colab --run-name myrun --epochs 25", shell=True, check=True)\n' \
  | colab --auth=adc exec -s cfd --timeout 3600

# Pull a run's artifacts back to this machine (optional — Drive is the source of truth)
printf 'import subprocess; subprocess.run("tar czf /content/cfd-sae-myrun.tgz -C /content/drive/MyDrive/cfd-sae checkpoints/myrun embeddings/myrun runs/myrun", shell=True, check=True)\n' \
  | colab --auth=adc exec -s cfd --timeout 300
colab --auth=adc download -s cfd /content/cfd-sae-myrun.tgz ./cfd-sae-myrun.tgz
tar xzf cfd-sae-myrun.tgz -C .

# Release the VM when done
colab --auth=adc stop -s cfd
```

### Dry-run / debugging

Every `colab.sh` command supports `--dry-run`, which prints the `colab`
commands it would run without executing them:

```bash
bash scripts/colab.sh --dry-run run train_mgn --run-name colab-run --epochs 25
bash scripts/colab.sh --dry-run download colab-run
```

### Legacy notebook flow (still works)

If you prefer the traditional notebook approach:

```python
# Cell 1: install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"

# Cell 2: clone + install
!git clone https://github.com/YOUR_USERNAME/cfd-sae /content/cfd-sae
%cd /content/cfd-sae
!uv sync

# Cell 3: run (example)
!uv run python scripts/train_mgn.py --hardware colab --run-name colab-smoke --epochs 1
```

---

## 9. Interactive Analysis (Notebooks)

After training (`train_mgn` → `extract_embeddings` → `train_sae`), explore the
SAE interactively in `notebooks/05_analysis.ipynb`.

### What the notebook does

The notebook loads an **existing** run's SAE checkpoint + embeddings and lets
you:
- Rank Top-K salient latents by the three Table-1 scores (Variance, MeanAbs, Entropy)
- Inspect any latent's activation histogram
- Visualize a latent's **spatial activation** on the most-activated frame

It does **not** train anything — it only analyzes existing artifacts.

### Running the notebook

**Local:**
```bash
# Start Jupyter
jupyter lab
# or: jupyter notebook

# Open notebooks/05_analysis.ipynb
# Set the run_name widget to your run (e.g., "smoke-run")
# Set the split widget (e.g., "test")
# Run all cells
```

**Colab:**
```bash
# Upload the .ipynb, or git clone + uv sync in a Colab runtime, then open it.
```

### Notebook workflow

1. **Cell 1 (Setup):** Imports `src.visualize`, `scripts.analyze`, and
   `src.data.cylinder_flow`. Prints `setup OK; repo: /path/to/repo`.

2. **Cell 2 (Load a run):** Displays `run_name` and `split` widgets. Loads all
   SAE codes and computes Table-1 saliency scores. Example output:
   ```
   split=test  samples=2307600  hidden=1024
     variance: max=0.8226  mean=0.1131
     mean_abs: max=0.4795  mean=0.2053
     entropy: max=3.4359  mean=2.1793
   ```

3. **Cell 3 (Top-K bars):** Renders three grouped bar charts — one per Table-1
   score — showing the top 20 most salient latents.

4. **Cell 4 (Interactive latent inspection):** Displays a slider (0 to 511).
   Sliding to a latent index shows:
   - A histogram of that latent's activations across all samples
   - A spatial scatter plot of the **most-activated frame** (node color = latent
     activation)

### Prerequisites for interactivity

The notebook requires `ipywidgets`, which is installed as a dev dependency by
`uv sync`. If you're running in Colab, install it first:
```python
!pip install ipywidgets
```

---

## 10. Configuration

### Hardware YAML presets

Each preset in `configs/hardware/` is a YAML file with these keys:

```yaml
# configs/hardware/desktop.yaml
batch_size: 1          # MGN batch size (always 1 — single graph per batch)
num_workers: 4        # DataLoader workers
pin_memory: true      # Pin memory for faster GPU transfer
mgn:
  hidden_dim: 128     # MGN hidden dimension (paper spec)
  message_passing_steps: 9  # Number of message-passing steps (paper spec)
  epochs: 50        # Default training epochs
sae:
  expansion: 8        # SAE expansion factor (hidden = input_dim * 8)
  lambda_l1: 3.0e-4   # L1 sparsity penalty coefficient
  batch_size: 256     # SAE training batch size
  lr: 1.0e-4          # SAE learning rate
```

### CLI overrides

Any preset value can be overridden on the command line:
```bash
# Override MGN epochs and SAE learning rate
uv run python scripts/train_mgn.py --hardware desktop --run-name my-run --epochs 100
uv run python scripts/train_sae.py --hardware desktop --run-name my-run --epochs 100
```

The `--epochs` flag overrides `config.mgn.epochs` or `config.sae.epochs`
depending on the script.

### Base directory (`--base-dir`)

All scripts accept a `--base-dir <path>` flag that overrides where
`data/`, `checkpoints/`, `embeddings/`, and `runs/` live. Default is the repo
root (or the mounted Drive path on Colab, which `scripts/colab.sh` already
handles for you). Useful for pointing runs at external/fast storage:

```bash
# Run all stages under a custom base directory
uv run python scripts/train_mgn.py --hardware desktop --run-name my-run \
  --epochs 50 --base-dir /mnt/fast-ssd/cfd-sae
```

Artifacts then appear under `/mnt/fast-ssd/cfd-sae/checkpoints/my-run/`,
`.../embeddings/my-run/`, and `.../runs/my-run/`.

No environment variables are used by the codebase; configuration is
flags-only.

---

## 11. Artifact Layout

For a run named `my-run`, the full artifact tree looks like:

```
checkpoints/my-run/
├── epoch_0001.pt          # Periodic checkpoint (last 3 retained)
├── epoch_0010.pt
├── epoch_0050.pt          # Latest
├── best.pt                # Best-loss checkpoint (MGN)
└── sae/
    ├── epoch_0001.pt      # SAE periodic checkpoints
    ├── epoch_0010.pt
    ├── best.pt            # Best train-loss SAE checkpoint
    └── best_val.pt        # Best validation-loss SAE checkpoint

embeddings/my-run/
├── test/
│   ├── ex00000_fr0000.npy   # [N, 128] node embeddings per (example, frame)
│   ├── ex00000_fr0001.npy
│   ├── ex00001_fr0000.npy
│   └── ...
├── embedding_stats.npz      # Per-feature mean/std (z-score normalization)

runs/my-run/
├── resolved_config.yaml     # Full resolved hardware config
├── run_metadata.json        # Timestamp, git, env, args, config snapshot
└── analysis.json            # Full analysis results (analyze stage)
```

### Checkpoint format

Each `.pt` checkpoint is a dict with:
```python
{
    "global_step": 1200,      # Total optimizer steps
    "epoch": 10,              # Current epoch
    "best_loss": 0.004567,    # Best training loss so far
    "best_val": 0.005678,     # Best validation loss (SAE only)
    "model_state": {...},     # PyTorch model state_dict
    "optimizer_state": {...}, # PyTorch optimizer state_dict
    "rng": null,              # RNG state (reserved for future use)
}
```

### Checkpoint retention

- The last 3 periodic checkpoints are retained (`keep_last=3` in
  `src/utils/checkpoint.py`). Older ones are deleted.
- `best.pt` is always kept (overwritten when a new best is found).
- For SAE, `best_val.pt` is also kept (best validation-loss weights, restored
  at the end of training per the paper).

---

## 12. Troubleshooting

### Common issues and fixes

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch_scatter'` | PyG sparse wheels are platform-specific | `uv pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html` (see [README notes](README.md#notes-on-pyg-sparse-wheels)) |
| `FileNotFoundError: Missing TFRecord for split 'test'` | Data not downloaded | `uv run python scripts/download_data.py --data-dir data --skip-existing` |
| `ValueError: Resume config mismatch for keys: mgn` | Config changed since last checkpoint | Use a new `--run-name`, or restore the original config values |
| `--resume requires an explicit --run-name` | Auto-generated run name can't be resumed | Pass `--run-name <name>` |
| `RuntimeError: CUDA out of memory` | Batch too large for GPU | Reduce `--max-steps` or use `--hardware colab` (smaller batch sizes) |
| `ModuleNotFoundError: No module named 'ipywidgets'` | Missing dev dependency | `uv sync` (installs dev deps) or `uv pip install ipywidgets` |
| `FileNotFoundError: No SAE checkpoint in ...` | SAE not trained yet | Run `train_sae.py` before `analyze.py` |
| `FileNotFoundError: No MGN checkpoint in ...` | MGN not trained yet | Run `train_mgn.py` before `extract_embeddings.py` |
| `google-colab-cli: command not found` | CLI not installed | `uv tool install google-colab-cli` |
| Colab auth expired | OAuth2 token expired | Re-authenticate: `colab login` |
| Drive permission denied | Drive not mounted or wrong path | Mount Drive first: `from google.colab import drive; drive.mount('/content/drive')` |

### Verifying a full smoke run

To verify the pipeline works end-to-end:

```bash
# 1. Download data
uv run python scripts/download_data.py --data-dir data --skip-existing

# 2. Train MGN (1 epoch, small)
uv run python scripts/train_mgn.py --hardware macbook --run-name verify-smoke --epochs 1

# 3. Extract embeddings (first 10 examples only)
uv run python scripts/extract_embeddings.py --hardware macbook --run-name verify-smoke --split test --max-examples 10

# 4. Train SAE (100 steps, no validation)
uv run python scripts/train_sae.py --hardware macbook --run-name verify-smoke --epochs 1 --max-steps 100 --no-val

# 5. Analyze
uv run python scripts/analyze.py --hardware macbook --run-name verify-smoke --split test --top-k 5
```

Expected: all stages complete without errors, and `runs/verify-smoke/analysis.json`
is created with valid results.

### Checking the resolved config

To see what config a run resolved to:
```bash
cat runs/<run-name>/resolved_config.yaml
```

To see full run metadata (git commit, env, args):
```bash
cat runs/<run-name>/run_metadata.json
```

### Running the linter

```bash
uv run ruff check src/ scripts/
```

---

## See Also

- **[README.md](../README.md)** — High-level project overview and quickstart
- **[docs/plans/phase-1-foundation.md](phase-1-foundation.md)** — Repository scaffolding and environment resolution
- **[docs/plans/phase-2-data-and-models.md](phase-2-data-and-models.md)** — Data loading and model implementations
- **[docs/plans/phase-3-training-pipeline.md](phase-3-training-pipeline.md)** — Training scripts and resume guarantees
- **[docs/plans/phase-4-colab-and-validation.md](phase-4-colab-and-validation.md)** — Colab workflow and validation matrix
- **[arXiv:2507.16069](https://arxiv.org/abs/2507.16069)** — The paper this pipeline implements