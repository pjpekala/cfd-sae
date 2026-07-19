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

## Quickstart (Colab)

In a fresh Colab runtime, install `uv`, sync deps, and run scripts with
`uv run`.

```python
# Cell 1: install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh

import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
```

```python
# Cell 2: clone + install
!git clone https://github.com/YOUR_USERNAME/cfd-sae /content/cfd-sae
%cd /content/cfd-sae
!uv sync
```

```python
# Cell 3: run (example)
!uv run python scripts/train_mgn.py --hardware colab --run-name colab-smoke --epochs 1
```

If you want data/checkpoints on Drive, set `CFD_SAE_BASE_DIR` to a mounted Drive
path before running scripts.

## Hardware Presets

Hardware preset YAMLs live in `configs/hardware/`:

- `colab.yaml`
- `desktop.yaml`
- `macbook.yaml`

All scripts support:

- `--hardware {auto,colab,desktop,macbook}`
- `--run-name <name>`
- `--seed <int>`
- `--resume` (except analysis)

`auto` resolves to:

- `colab` when running in Colab
- `macbook` on Darwin hosts
- `desktop` otherwise

## Run Isolation and Resume Contract

Every run is isolated under a run name.

- Checkpoints: `checkpoints/<run_name>/`
- Embeddings: `embeddings/<run_name>/`
- Metadata/config snapshot: `runs/<run_name>/`

Resume safety behavior:

- `--resume` requires explicit `--run-name` (or `CFD_SAE_RUN_NAME`).
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

## Optional Env Vars

See `.env.example`:

- `CFD_SAE_RUN_NAME`
- `CFD_SAE_HARDWARE`
- `CFD_SAE_BASE_DIR`

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
