# Phase 4 - Colab Workflow and Validation

Finalize Colab-first usability and reproducibility validation. The chosen
backend is **google-colab-cli** (terminal CLI), which replaces the notebook
setup-cell workflow: it provisions a GPU VM, runs our scripts via `uv run`, and
tears down. Our scripts are unchanged (`--hardware colab` + `uv run` contract).

## Goals

- Make Colab bring-up fast, repeatable, and notebook-free.
- Validate restart/resume across Colab runtime resets.
- Document the CLI workflow in README.

## Deliverables

- `scripts/colab_run.sh` — provision + upload + exec + download + stop wrapper.
- `google-colab-cli` in dev dependency group.
- Updated `README.md` Colab quickstart (CLI primary; legacy notebook flow in details).
- Final reproducibility checklist documented.

## Tasks

1. `scripts/colab_run.sh`: drives `colab new --gpu` -> `colab upload` ->
   `colab exec "uv run python scripts/<stage>.py --hardware colab ..."` ->
   `colab download checkpoints embeddings runs ./` -> `colab stop`.
   Env: `COLAB_GPU` (default A100), `COLAB_KEEP=1` (leave VM for follow-up).
2. Drive persistence: set `CFD_SAE_BASE_DIR` to a `colab drivemount` path so
   checkpoints/embeddings survive teardown; `--resume` continues on a later VM.
3. README Colab quickstart (CLI primary) + troubleshooting (PyG wheel, missing
   TFRecord pkg, Drive perms, OOM/batch fallback).
4. Notebooks (OPTIONAL): if desired, thin wrappers that call the same scripts.
   Deprioritized now that colab-cli covers the workflow without notebooks.
5. Validation matrix (local dry-run + real Colab):
   - Fresh VM: `colab_run.sh train_mgn` writes checkpoint + downloads it.
   - Runtime reset: relaunch with `--resume` continues.
   - Downstream: extract -> train_sae -> analyze all produce artifacts.

## Validation Matrix

1. Local dry-run of `colab_run.sh` (no GPU): confirm command construction.
2. Fresh Colab VM: `train_mgn.py` checkpoints + downloads to ./checkpoints.
3. After reset: `--resume` restores and continues.
4. Full pipeline outputs validated end-to-end.

## Acceptance Criteria

- A new Colab session can run the full pipeline with only `colab_run.sh <stage>`.
- README instructions sufficient for another engineer to reproduce.
- Resume across resets confirmed.

## Risks and Mitigations

- Risk: CLI auth/expiry.
  - Mitigation: `colab` uses OAuth2 by default; document re-auth.
- Risk: Large upload/download of repo + artifacts.
  - Mitigation: `.gitignore`-style excludes (venv/data) already in place; only
    upload source, download checkpoints/embeddings/runs.
- Risk: Notebook/script drift.
  - Mitigation: notebooks (if added) call shared script functions, not copies.

## Exit Checklist

- [x] `scripts/colab_run.sh` wrapper implemented + bash syntax checked.
- [x] `google-colab-cli` added to dev deps; `uv sync` installs it.
- [x] README Colab quickstart updated (CLI primary).
- [ ] Real Colab VM run validated (fresh + resume).
- [ ] Full pipeline outputs validated end-to-end on Colab.
- [ ] (Optional) notebooks parity.
