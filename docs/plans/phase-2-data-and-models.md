# Phase 2 - Data and Models

Implement minimal, reliable data/model components required for pipeline
execution. Focus on correctness and clear interfaces, not benchmark tuning.

## Goals

- Load cylinder-flow data consistently across splits.
- Implement baseline MGN and SAE modules with stable APIs.
- Persist normalization/stat artifacts needed by downstream stages.

## Deliverables

- `src/data/cylinder_flow.py`
- `src/models/mgn.py`
- `src/models/sae.py`
- `src/utils/checkpoint.py`
- Unit-like smoke scripts or quick checks embedded in training scripts

## Tasks

1. Implement dataset loader in `src/data/cylinder_flow.py`:
   - Split handling (`train`, `valid`, `test`).
   - TFRecord reading/parsing.
   - Conversion into tensors/graph objects expected by MGN.
   - Basic schema validation with actionable errors.
2. Implement normalization/stat handling:
   - Compute training stats.
   - Save to `run_dir` and/or shared artifact file.
   - Reuse exact stats in extraction/analysis.
3. Implement `src/models/mgn.py`:
   - Config-driven constructor.
   - `forward(...)` returns predictions and optional internals.
   - `loss(...)` or helper for training objective.
   - Shape assertions at key boundaries.
4. Implement `src/models/sae.py`:
   - Config-driven constructor with expansion factor.
   - Reconstruction + sparsity loss (`lambda_l1`).
   - Methods suitable for batched embedding tensors.
5. Implement checkpoint utility in `src/utils/checkpoint.py`:
   - `save_checkpoint(state, path, atomic=True)`.
   - `load_latest(run_ckpt_dir, model, optimizer, scheduler, scaler)`.
   - Save/load RNG states for reproducibility.
   - Retain last N checkpoints and `best.pt`.

## Acceptance Criteria

- A tiny train loop can consume one batch from each split without crashing.
- MGN forward pass and loss compute successfully on one batch.
- SAE forward pass and loss compute on dummy extracted embeddings.
- Checkpoint save/load restores model + optimizer + epoch + global step.

## Risks and Mitigations

- Risk: TFRecord parsing mismatches expected field keys/shapes.
  - Mitigation: Validate and fail fast with clear key-level errors.
- Risk: Device issues on MPS/CPU.
  - Mitigation: Keep default worker/pin_memory conservative per preset.

## Exit Checklist

- [ ] Data loader supports all three splits.
- [ ] Stats persistence implemented.
- [ ] MGN/SAE constructors and forwards pass smoke tests.
- [ ] Checkpoint utility restores full training state.
