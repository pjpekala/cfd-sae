# Phase 4 - Colab Workflow and Validation

Finalize Colab-first usability, notebook parity, and reproducibility validation.

## Goals

- Make fresh Colab bring-up fast and repeatable.
- Ensure notebook workflow mirrors script behavior.
- Validate restart/resume across Colab runtime resets.

## Deliverables

- Notebooks:
  - `notebooks/01_train_mgn.ipynb`
  - `notebooks/02_extract_embeddings.ipynb`
  - `notebooks/03_train_sae.ipynb`
  - `notebooks/04_analysis.ipynb`
- Updated `README.md` with Colab quickstart and troubleshooting
- Final reproducibility checklist results documented

## Tasks

1. Add common notebook setup cell:
   - Detect Colab.
   - Mount Drive.
   - Clone/pull repo.
   - Install dependencies.
   - Construct env via `get_env(HARDWARE)`.
2. Keep notebooks thin:
   - Prefer invoking underlying script/module functions.
   - Avoid duplicate logic in notebook cells.
3. Write Colab quickstart in README:
   - First-time setup.
   - Resume workflow after disconnect.
   - How to switch/lock `run_name`.
4. Add troubleshooting:
   - PyG wheel compatibility.
   - Missing TFRecord package.
   - Drive path permissions.
   - OOM and batch-size fallback.
5. Perform validation matrix:
   - Fresh runtime run.
   - Runtime reset.
   - Resume run.
   - Continue to embeddings/SAE/analysis.

## Validation Matrix

1. Fresh Colab runtime:
   - Setup cell succeeds.
   - `train_mgn.py` writes checkpoint.
2. After runtime reset:
   - Setup cell succeeds again.
   - Resume picks latest checkpoint and continues.
3. Downstream stages:
   - Embeddings generated under run-scoped directory.
   - SAE trains and checkpoints.
   - Analysis outputs generated.

## Acceptance Criteria

- A new Colab session can recover and continue progress with only:
  - setup cell
  - one resume command
- README instructions are sufficient for another engineer to reproduce.

## Risks and Mitigations

- Risk: Notebook/script drift.
  - Mitigation: Notebook calls shared script functions.
- Risk: Dependency issues in Colab image changes.
  - Mitigation: Pin critical package versions and document alternate wheels.

## Exit Checklist

- [ ] All four notebooks run with shared setup logic.
- [ ] README Colab quickstart verified in a fresh session.
- [ ] Resume after reset confirmed.
- [ ] Full pipeline outputs validated end-to-end.
