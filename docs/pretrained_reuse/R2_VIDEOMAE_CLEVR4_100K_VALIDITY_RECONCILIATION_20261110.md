# Clevr-4 VideoMAE Path A result validity reconciliation — 2026-11-10

## Why this note exists

The completed fit report and the preceding exposure audit use different
dispositions for the internal-validation pool. This note preserves both
records and states the current claim boundary; it does not rerun inference or
change any result.

## Chronology and evidence

1. The frozen-input/identity audit at
   [`R2_CLEVR4_100K_INTERNAL_VALIDATION_EXPOSURE_AUDIT_20261110.md`](R2_CLEVR4_100K_INTERNAL_VALIDATION_EXPOSURE_AUDIT_20261110.md)
   states that the candidate pool was **not yet admitted for Decision-quality
   evaluation**. It found zero exact-byte and canonical-pixel matches against
   the preserved 10k images, but the synthetic-scene pHash threshold was
   uncalibrated and the cross-worktree exposure scan was incomplete. It
   conservatively quarantined the 1,706 rows with pHash distance <=4 or prior
   ID-string collisions, leaving a 6,296-row candidate pool. Its next gate
   explicitly required identity/exposure review before scoring.
2. The later fixed experiment used 72,010 training images, 1,000 selector
   images, and a 5,296-image one-time reporting pool, and completed eight
   epochs. The source-run report is
   [`R2_VIDEOMAE_CLEVR4_100K_PATH_A_RESULTS_20261110.md`](R2_VIDEOMAE_CLEVR4_100K_PATH_A_RESULTS_20261110.md);
   its code/config source commit is `01d37d3c0350cc254407542ab1f10f7373d4d8ca`.
   Within-run train/selector/report partitions have zero ID, media-hash, and
   asset overlap. No official validation or sealed-audit records were loaded.
3. The report records 79.62% equal-taxonomy selector accuracy and 80.01% on the
   one-time reporting pool. Reporting labels were read after the epoch-8
   checkpoint was selected and reloaded, so those labels did not select this
   checkpoint. However, the earlier admission gate was not documented as
   cleared before the run, and the full cross-worktree exposure review and
   calibrated synthetic-scene near-duplicate test remain unproven.

## Current disposition

Preserve the fitted checkpoint and all reported predictions as historical
artifacts. Treat the 80.01% result as an **observed, one-time internal
development measurement with unresolved prior-exposure risk**, not as a clean
or blind independent validation result. Do not use its reporting pool for
future training, architecture selection, threshold choice, or promotion. Do
not describe the earlier pre-run audit's "not yet admitted" statement as if it
had been resolved. The selector result remains development evidence as well;
it was used for checkpoint selection.

The per-taxonomy result remains useful for diagnosis: reporting Accuracy was
98.70% color, 79.04% texture, 43.43% count, and 98.89% shape. These figures
are task-scoped observations under the above limitation, not a pass of the
roadmap's image quality gate. Count and texture remain weak. The model is
static-image only; repeating a still across VideoMAE frames is not temporal
video evidence.

No new evaluation should use this Clevr-4 pool. Any future image-quality gate
needs a newly admitted source/task cohort with verified rights, source-group
and normalized-content separation, and a complete prior-exposure scan before
model selection. The original 10k and official validation results stay
unchanged; no sealed audit was opened.
