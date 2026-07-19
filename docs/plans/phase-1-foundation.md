# Phase 1 - Foundation

Establish repository structure, environment resolution, and config plumbing so
all later phases share one consistent runtime contract.

## Goals

- Create project scaffolding for `src`, `configs`, `scripts`, and `notebooks`.
- Implement `src/env.py` with hardware auto-detection and explicit run paths.
- Add hardware YAML presets and a shared loader.
- Define run naming and isolated artifact directory strategy.

## Deliverables

- Directory tree:
  - `configs/hardware/{colab,desktop,macbook}.yaml`
  - `src/env.py`
  - `src/config.py` (or equivalent loader)
  - `src/utils/{seed.py,io.py}` (minimal reproducibility helpers)
  - `scripts/` placeholders with argument parser stubs
- Package init files:
  - `src/__init__.py`
  - `src/data/__init__.py`
  - `src/models/__init__.py`
  - `src/utils/__init__.py`
- Base docs:
  - `README.md` skeleton with install and quickstart placeholders
  - `.env.example`
  - `requirements.txt`

## Tasks

1. Create missing directories and `__init__.py` files.
2. Implement `Env` dataclass in `src/env.py` with fields:
   - `hardware`, `device`, `root`, `data_dir`, `ckpt_dir`, `embed_dir`,
     `run_name`, `run_dir`, `on_colab`.
3. Implement `get_env(hardware="auto", run_name=None)`:
   - Detect Colab.
   - Resolve hardware preset.
   - Resolve `device` safely (`cuda`, `mps`, `cpu`).
   - Build run-isolated paths:
     - `checkpoints/<run_name>`
     - `embeddings/<run_name>`
   - Create directories.
4. Implement config loading:
   - Load `configs/hardware/<preset>.yaml`.
   - Merge CLI overrides if provided.
   - Persist resolved config snapshot to run directory.
5. Define `run_name` policy:
   - Default format: `<stage>-<YYYYMMDD-HHMMSS>-<hardware>-s<seed>`.
   - Allow explicit override from CLI.
6. Add script parser stubs with uniform flags:
   - `--hardware`, `--run-name`, `--seed`, `--resume`.

## Acceptance Criteria

- `python scripts/train_mgn.py --hardware macbook --run-name smoke` starts,
  prints resolved environment/config, and exits cleanly.
- Running twice with same args resolves to same run directory if `--run-name`
  is explicit.
- Directory creation is idempotent and safe when rerun.

## Risks and Mitigations

- Risk: Path confusion between repo-local and Drive paths in Colab.
  - Mitigation: Centralize all path logic in `get_env()` only.
- Risk: Script drift (different parser flags).
  - Mitigation: Shared CLI helper reused by all scripts.

## Exit Checklist

- [ ] All phase directories/files exist.
- [ ] `env.hardware` available and tested.
- [ ] Hardware YAMLs load successfully.
- [ ] Resolved config snapshot persisted per run.
- [ ] Uniform script flags implemented.
