# Previous-command probability baseline on fixed DROID train folds

Freeze this train-only diagnostic before computing probability metrics. Reuse
the existing 275-record train-only previous-command predictions and the exact
five outer recording folds and saved raw/delta sensor readout predictions from
the pinned temporal comparison. Require exact IDs, groups, targets, row order,
hashes, and one held fold per example. Read no validation, audit, future-action
data, source media or encoders.

For each held fold fit a seven-by-seven previous-command-to-next-target count
table on only its four training folds. Apply add-one smoothing to all seven
outcomes for each previous command. Report all seven-class Accuracy, macro
Accuracy, NLL, macro NLL, multiclass Brier, 10-bin ECE and per-class counts via
the pinned prior readout's metric function. Also report matched sensor-only
raw/delta metrics and, separately, stable-command and changed-command subsets.
The latter are descriptive subsets defined retrospectively by their targets;
they are not filters for a held-out benchmark. No alpha or other tuning, model
selection, new hyperparameter search, validation or final evaluation is allowed.

Previous command is an extra control input unavailable to the compared sensor
readouts, so this is a task-persistence baseline rather than an architecture
comparison. The 275 recordings come from an observed training cohort, and
recording-group folds do not establish independent operators or sites. High
accuracy may simply reflect temporal persistence and is not visual, multimodal,
or product-quality evidence. Save all OOF probabilities and inputs privately;
do not overwrite existing artifacts or create a product checkpoint.
