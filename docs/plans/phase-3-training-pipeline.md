# Phase 3 - Training Pipeline and Resume Guarantees

Build end-to-end scripts with strong resume behavior and reproducibility
metadata. This phase makes the pipeline operational.

## Goals

- Implement all CLI scripts with a shared pattern.
- Guarantee resume correctness for interrupted runs.
- Produce run-scoped artifacts for every stage.

## Deliverables

- `scripts/download_data.sh`
- `scripts/train_mgn.py`
- `scripts/extract_embeddings.py`
- `scripts/train_sae.py`
- `scripts/analyze.py`

## Shared CLI Contract

All Python scripts should support a consistent core:

- `--hardware {auto,colab,desktop,macbook}`
- `--run-name <string>`
- `--seed <int>`
- `--resume`

Optional stage-specific limits for smoke tests:

- `--epochs`, `--max-steps`, `--max-batches`, `--split`

## Tasks

1. `download_data.sh`:
   - Download train/valid/test TFRecords.
   - Resume partial downloads safely.
   - Print final file list and sizes.
2. `train_mgn.py`:
   - Initialize env/config/seed.
   - Create run metadata file (`run.json` or yaml).
   - Train loop with periodic checkpointing.
   - Resume support restoring full state.
3. `extract_embeddings.py`:
   - Load MGN checkpoint (`best` or `latest`).
   - Stream through requested split.
   - Save embeddings in chunked format by run.
4. `train_sae.py`:
   - Load extracted embeddings from selected run.
   - Train with checkpointing and resume.
   - Save best and latest checkpoints.
5. `analyze.py`:
   - Load SAE checkpoint and embedding artifacts.
   - Compute basic diagnostics (sparsity, recon metrics, top features).
   - Save outputs in run-scoped analysis directory.
6. Add config compatibility checks on resume:
   - Compare saved resolved config to current resolved config.
   - Warn/fail on critical mismatches (model dims, data paths, etc.).

## Acceptance Criteria

- Forced interruption test:
  - Kill training mid-run.
  - Relaunch with `--resume`.
  - Verify resumed epoch/global_step and continued loss logging.
- End-to-end local smoke:
  - Train MGN (small limits) -> extract embeddings -> train SAE -> analyze.
- Run artifacts are isolated under each `run_name`.

## Risks and Mitigations

- Risk: Silent config drift on resume.
  - Mitigation: Persist resolved config and enforce compatibility checks.
- Risk: Large embedding writes causing memory pressure.
  - Mitigation: Chunked writing and periodic flush.

## Exit Checklist

- [x] All scripts share core CLI flags.
- [x] Resume path tested under interruption (kill mid-run -> `--resume` continues from saved global_step; verified global_step 595 -> 600+).
- [x] Metadata and config snapshots saved per run (`resolved_config.yaml`, run metadata).
- [x] End-to-end smoke run successful (train -> extract[test] -> train_sae -> analyze, all on real data).

## Implementation Notes (2026-07-19)

- `train_mgn.py`: full epoch loop over valid split, periodic (`--save-every`) + best
  checkpointing, `--resume` restores model+optimizer+global_step (verified).
- `extract_embeddings.py`: loads MGN `best.pt`, writes pre-decoder node embeddings
  h_i (paper's SAE input) as chunked `.npy` per (example,frame) under
  `embeddings/<run>/<split>/`. Defaults to test split per paper.
- `train_sae.py`: trains SparseAutoencoder with decoder unit-L2-norm (paper §3.2),
  batch=128, lambda=3e-4, checkpointing + `--resume`.
- `analyze.py`: recon MSE, mean-L1, frac-inactive, and Top-K salient latents by the
  three Table-1 scores (Variance, MeanAbs, Entropy).
- `src/models/sae.py`: added `normalize_decoder()` (after init + every step).
- `src/models/mgn.py`: added `node_embeddings()`; MLPs now use LayerNorm + 2 hidden
  layers (paper §4.2).
- `src/utils/io.py`: added `read_json`/`read_yaml` + `_require_yaml`.
