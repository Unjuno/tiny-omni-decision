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

- CPU regression tests cover sample-ID order, group-ID order, target order, and
  missing per-arm identity mappings.
- Ruff, focused pytest, repository CPU tests, and GitHub Actions status are to
  be recorded for the implementation commit.
- No model, media, validation cohort, or sealed audit was loaded. This is an
  evaluation-contract change, not new modality-quality evidence; R2-M still
  lacks an admitted real task requiring multiple inputs.
