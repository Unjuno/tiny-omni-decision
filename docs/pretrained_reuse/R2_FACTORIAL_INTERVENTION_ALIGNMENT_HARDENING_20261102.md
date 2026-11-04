# Factorial intervention arm-alignment hardening — 2026-11-02

## Change

The saved-prediction evaluator now requires every prediction arm to provide
its ordered sample IDs, source-group IDs, and target indices. Each arm must
match the canonical rows exactly, including order; missing identity maps,
permutations, and label mismatches fail closed before metrics or paired group
intervals are computed. The report schema is now version 2 and includes a
SHA-256 for each arm's ordered sample, group, and target identities.

The prior evaluator accepted one shared identity vector for all arms. It could
therefore not detect a caller that reordered one arm's predictions while
passing the unchanged shared IDs. This follow-up closes that evaluator-level
integrity gap. It does not prove that upstream predictions were generated from
the claimed input intervention, nor that the task requires the changed
modality.

## Verification and limits

- CPU regression tests cover sample-ID order, group-ID order, target order,
  missing per-arm identity mappings, and probability validation after identity
  alignment. Direct Python smoke scenarios and Ruff passed locally.
- GitHub Actions for implementation commit `e0b87253` passed on both push and
  pull-request events, including repository-wide `pytest -q`, Ruff, and
  manifest validation: [push run](https://github.com/Unjuno/tiny-omni-decision/actions/runs/41382108611),
  [pull-request run](https://github.com/Unjuno/tiny-omni-decision/actions/runs/41382115658).
- Local pytest was not invoked because Windows Application Control blocks the
  `pytest.exe` launcher; that policy was not bypassed.
- No model, media, validation cohort, or sealed audit was loaded. This is an
  evaluation-contract change, not new modality-quality evidence; R2-M still
  lacks an admitted real task requiring multiple inputs.
