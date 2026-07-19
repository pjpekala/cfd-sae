# CFD-SAE Implementation Plans

This folder contains phase-by-phase implementation plans for a colab-first,
resumable, reproducible pipeline.

## Locked Decisions

- Priority: get a reliable end-to-end pipeline running in Colab first.
- Run isolation: store artifacts under run-specific directories
  (`checkpoints/<run_name>/...`, `embeddings/<run_name>/...`).
- Quality metrics are secondary in v1; reliability and restartability are
  primary.

## Plan Files

- `docs/plans/phase-1-foundation.md`
- `docs/plans/phase-2-data-and-models.md`
- `docs/plans/phase-3-training-pipeline.md`
- `docs/plans/phase-4-colab-and-validation.md`

## Recommended Execution Order

1. Complete Phase 1 fully before coding model internals.
2. Complete Phase 2 with small smoke runs only.
3. Complete Phase 3 and verify resume behavior under forced interruption.
4. Complete Phase 4 in Colab and document the exact workflow.

## Definition of Success for v1

- From a fresh Colab runtime, you can train, checkpoint, disconnect/restart,
  resume, extract embeddings, train SAE, and run analysis.
- Each run has self-contained metadata and checkpoints for reproducibility.
- Same `--seed` + same resolved config + same checkpoint produces consistent
  continuation behavior.
