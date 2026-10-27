# Set-decision error decomposition

This is a post-hoc descriptive analysis of the previously observed 2,593
train-source out-of-fold joint predictions. It checks the pinned receipts,
prediction hashes, IDs and targets; no new inference, fit or audit was run.

| True label | Total | Exact | Chose none despite a visible target | False visible when none | Missing only | Extra only | Both missing and extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| None visible | 692 | 366 | — | 326 | — | — | — |
| One target | 799 | 188 | 75 | — | 0 | 95 | 441 |
| Multiple targets | 1,102 | 68 | 33 | — | 492 | 73 | 436 |

For multiple targets, 928/1,102 predictions omitted at least one true target
while selecting some visible candidate; a further 33 selected none. This is a
strong reason to inspect candidate recall and threshold behavior, not proof
that a particular loss, backbone or data source is the sole cause. All errors
are counted once. No threshold or checkpoint was tuned in this analysis.

Paired to the previous single-choice model at seven epochs, the new model was
correct where the old was wrong in 68 multi-target, 56 single-target and 27
none-visible examples. The old model alone was correct in 140 single-target
and 128 none-visible examples; it cannot exactly match any multi-target set.
Thus the increased ability to emit sets recovered some multi-target cases but
did not compensate for losses on the other two strata. These changes involve
a different loss and decision rule and cannot isolate either effect.

The conditional scores are not a normalized joint set distribution, these
folds are nonblind and the common source may share contributors across folds.
The next method comparison must use fixed training-only threshold selection,
retain an untouched group for evaluation and report all three strata as well
as aggregate calibration; it cannot claim independent transfer until source
identity is established. The current analysis does not select a model.

Output: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-set-error-audit-20261027-attempt-01/result.json`,
SHA-256 `8d09eef977b6046027a3559b607d2ed1fbd817fb5675ced853741c8fff4bd501`.
