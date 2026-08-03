## CFD-SAE (Colab-First, Resumable Pipeline)

This repo is being built as a reproducible, resumable pipeline for:

1. training an MGN model,
2. extracting embeddings,
3. training an SAE,
4. running interpretability analysis.

Current priority is reliability and restartability (especially on Colab), not
benchmark tuning.

## Environment Management (uv-first)

This project uses `uv` for all dependency and Python environment management.

- Dependency source of truth: `pyproject.toml` + `uv.lock`
- Managed virtualenv: `.venv`
- Recommended command pattern: `uv run python ...`

Do not rely on system `pip` in this repo.

## Quickstart (Local)

Prerequisites:

- `uv` installed: https://docs.astral.sh/uv/
- Python version from `.python-version` (currently `3.12`)

Install dependencies:

```bash
uv sync
```

Smoke check CLI wiring:

```bash
uv run python scripts/train_mgn.py --help
uv run python scripts/train_mgn.py --hardware auto --run-name smoke-local --epochs 1
```

## Quickstart (Colab via google-colab-cli)

`google-colab-cli` manages a **persistent, Drive-mounted** GPU VM. Code is synced
via git and artifacts live on Google Drive, so stages chain naturally and
`--resume` works across sessions. No notebooks required.

```bash
# 1) Install the CLI (one time) + authenticate
uv tool install google-colab-cli
#    Authenticate the CLI (see "Authenticating the Colab CLI" below). The mounted
#    Drive belongs to the Google account the CLI authenticates with.

# 2) Provision + mount Drive (one time per VM)
bash scripts/colab.sh new          # T4 GPU (free tier); --gpu L4/A100/CPU to change
bash scripts/colab.sh drive        # approve in browser, press Enter
#    Verify which account owns the mount first:  colab whoami

# 3) Sync code + deps (clones/pulls this repo; run again after each git push)
bash scripts/colab.sh sync

# 4) Run pipeline stages against the VM (artifacts persist on Drive)
bash scripts/colab.sh run train_mgn          --run-name myrun --epochs 25
bash scripts/colab.sh run extract_embeddings --run-name myrun --split test
bash scripts/colab.sh run train_sae          --run-name myrun --epochs 25
bash scripts/colab.sh run analyze            --run-name myrun --split test

# 5) Pull a run's artifacts back to this machine (optional — Drive is the source of truth)
bash scripts/colab.sh download myrun

# 6) Release the VM when done
bash scripts/colab.sh stop
```

The repo URL is taken from your local `git remote get-url origin` (ssh/git URLs
are converted to https for the VM's anonymous clone). Override with `--repo-url`
if needed. Use `--dry-run` to preview any command without executing it.

## Authenticating the Colab CLI

The Colab CLI authenticates to Google with **your** account, and **that account's
Google Drive is the one mounted** by `colab.sh drive` (not the VM's account).
Check which account is active:

```bash
colab whoami          # hidden debug cmd: prints active email, scopes, expiry
```

> `colab auth` is unrelated to CLI authentication — it injects *VM-side* GCP
> credentials (BigQuery/GCS). Don't use it to "log in" the CLI.

Two auth strategies exist; the flag must come **before** the subcommand.

### oauth2 (CLI default)

One-time browser consent flow; the token is cached at
`~/.config/colab-cli/token.json`. To use a different Google account, clear the
cache and re-run any colab command (pick the account in the browser):

```bash
rm ~/.config/colab-cli/token.json
colab whoami
```

If a VM was already provisioned under the old account, `colab stop -s <name>`
it first — it stays reachable only under that account.

### adc (Application Default Credentials)

Useful when the target Drive lives in a specific account (e.g., a student
account with more space). Authenticate gcloud ADC as that account with **all
four** scopes the Colab backend requires — a plain
`gcloud auth application-default login` is missing `colaboratory`, which makes
`colab new` 403 and unassign the fresh VM:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Then prefix every command with `--auth=adc`:

```bash
colab --auth=adc whoami
colab --auth=adc new -s cfd --gpu T4
colab --auth=adc drivemount -s cfd
```

`scripts/colab.sh` wraps `colab` with the default oauth2 auth and does not pass
`--auth`. If you authenticate via ADC, drive the VM with raw
`colab --auth=adc ...` commands instead — see "Using ADC (raw commands)" in
`docs/HOWTO.md` §8.

<details>
<summary>Legacy notebook flow (still works)</summary>

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
</details>

## Hardware Presets

Hardware preset YAMLs live in `configs/hardware/`:

- `colab.yaml` — also sets `gpu: T4` (the Colab VM GPU; override via `--gpu`)
- `desktop.yaml`
- `macbook.yaml`

All three use MGN `hidden_dim: 128` and 9 message-passing steps (paper spec).
Tune epochs, batch size, SAE `expansion`/`lambda_l1`/`lr` here.

Common script flags (all stages):

- `--hardware {auto,colab,desktop,macbook}`
- `--run-name <name>` (artifacts isolated per run)
- `--seed <int>`
- `--resume` (continues `train_mgn`/`train_sae`; needs explicit `--run-name`)
- `--epochs <int>` / `--max-steps <int>` (cap work for smoke tests)
- `--base-dir <path>` (override where data/checkpoints/embeddings live; default is repo root)
- `train_sae` only: `--val-frac`, `--patience`, `--min-epochs`, `--no-val`
  (see "SAE early-stopping" below)

`auto` resolves to:

- `colab` when running in Colab
- `macbook` on Darwin hosts
- `desktop` otherwise

## Command Reference (per machine)

The pipeline is four stages that chain on a shared `--run-name`:
`train_mgn` → `extract_embeddings` → `train_sae` → `analyze`.
Pick a stable run name so artifacts line up. `--resume` continues
`train_mgn`/`train_sae` (needs explicit `--run-name`).

### MacBook

```bash
uv sync
uv run python scripts/download_data.py --data-dir data --skip-existing   # once

uv run python scripts/train_mgn.py         --hardware macbook --run-name myrun --epochs 5
uv run python scripts/extract_embeddings.py --hardware macbook --run-name myrun --split test
uv run python scripts/train_sae.py         --hardware macbook --run-name myrun --epochs 50
uv run python scripts/analyze.py           --hardware macbook --run-name myrun --split test
```

### Personal PC (desktop preset)

```bash
uv sync
uv run python scripts/download_data.py --data-dir data --skip-existing   # once

uv run python scripts/train_mgn.py         --hardware desktop --run-name myrun --epochs 50
uv run python scripts/extract_embeddings.py --hardware desktop --run-name myrun --split test
uv run python scripts/train_sae.py         --hardware desktop --run-name myrun --epochs 50
uv run python scripts/analyze.py           --hardware desktop --run-name myrun --split test
```

### Colab (google-colab-cli — persistent Drive-mounted GPU VM)

```bash
uv tool install google-colab-cli                                         # once
# Authenticate: oauth2 (default) or adc — see "Authenticating the Colab CLI".

# Provision + mount Drive (one time per VM). Defaults to a T4 GPU (free-tier).
bash scripts/colab.sh new          # --gpu L4/A100/CPU to change
bash scripts/colab.sh drive        # approve in browser, press Enter

# Sync code + deps (clones/pulls this repo; re-run after each git push)
bash scripts/colab.sh sync

# Run pipeline stages. Artifacts persist on Drive, so stages chain via --run-name.
bash scripts/colab.sh run train_mgn          --run-name myrun --epochs 25
bash scripts/colab.sh run extract_embeddings --run-name myrun --split test
bash scripts/colab.sh run train_sae          --run-name myrun --epochs 25
bash scripts/colab.sh run analyze            --run-name myrun --split test

# Pull a run's artifacts back to this machine (optional — Drive is the source of truth)
bash scripts/colab.sh download myrun

# Release the VM when done
bash scripts/colab.sh stop
```

Every `colab.sh` command supports `--dry-run` to preview without executing.

### Notebooks (interactive analysis — no training)

`notebooks/05_analysis.ipynb` loads an **existing** run's SAE checkpoint +
embeddings and visualizes it (Top-K latents, histograms, spatial activation).
It does NOT train.

```bash
# Local
jupyter lab                      # or: jupyter notebook
# open notebooks/05_analysis.ipynb, set the run_name + split widgets, run all cells

# Colab
# upload the .ipynb (or git clone + uv sync in a Colab runtime), then open it.
```

Prereq for interactivity: `ipywidgets` (a dev dep, installed by `uv sync`).

### SAE early-stopping (paper-faithful)

`train_sae.py` holds out `--val-frac` (default 0.1) of the embedding files as a
validation set and stops when the held-out reconstruction MSE plateaus
(paper §4.2: "training proceeds until the reconstruction loss on a held-out
validation set stops decreasing"). The best-validation weights are restored at
the end. Tune with `--patience` (epochs w/o improvement, default 5),
`--min-epochs` (default 1), `--val-frac`, and `--no-val` to disable (fixed
epochs). Embedding z-score stats are computed over the train subset only.

### Hardware / GPU notes

- All presets now use MGN `hidden_dim: 128` (paper spec). The Colab preset
  requests `gpu: T4` by default (free-tier compatible); override with
  `bash scripts/colab.sh new --gpu L4` (or `--gpu A100` on Pro+), or
  `--gpu CPU` for a CPU runtime.

## Run Isolation and Resume Contract

Every run is isolated under a run name.

- Checkpoints: `checkpoints/<run_name>/`
- Embeddings: `embeddings/<run_name>/`
- Metadata/config snapshot: `runs/<run_name>/`

Resume safety behavior:

- `--resume` requires explicit `--run-name`.
- Resume validates key config compatibility against the saved snapshot.

## Data Download

Python (preferred with uv):

```bash
uv run python scripts/download_data.py --data-dir data --skip-existing
```

Shell alternative:

```bash
bash scripts/download_data.sh ./data
```

Expected files:

- `data/train.tfrecord`
- `data/valid.tfrecord`
- `data/test.tfrecord`

## Interactive Analysis (Notebooks)

After training (`train_mgn` → `extract_embeddings` → `train_sae`), explore the
SAE interactively in `notebooks/05_analysis.ipynb`:

- Rank Top-K salient latents by the three Table-1 scores (Variance, MeanAbs, Entropy).
- Inspect any latent's activation histogram.
- Visualize a latent's **spatial activation** on the most-activated frame.

Open it in Jupyter/Colab. It reuses `src/visualize.py` (shared plotting) and the
existing `scripts.analyze` functions — no duplicated logic. Set the `run_name`
widget to the run you want to explore.

## Script Entry Points (Phase-1 Scaffolding)

- `scripts/train_mgn.py`
- `scripts/extract_embeddings.py`
- `scripts/train_sae.py`
- `scripts/analyze.py`

Example command flow:

```bash
uv run python scripts/train_mgn.py --hardware auto --run-name smoke-run --epochs 1
uv run python scripts/extract_embeddings.py --hardware auto --run-name smoke-run --split train --max-batches 10
uv run python scripts/train_sae.py --hardware auto --run-name smoke-run --epochs 1 --max-steps 100
uv run python scripts/analyze.py --hardware auto --run-name smoke-run
```

## Notes on PyG Sparse Wheels

`torch_scatter` and `torch_sparse` can be platform-specific. Install only when
needed:

```bash
uv pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

For local Mac dev/smoke tests, these are often unnecessary.

## Planning Docs

Implementation plans are tracked in `docs/plans/`:

- `docs/plans/phase-1-foundation.md`
- `docs/plans/phase-2-data-and-models.md`
- `docs/plans/phase-3-training-pipeline.md`
- `docs/plans/phase-4-colab-and-validation.md`
