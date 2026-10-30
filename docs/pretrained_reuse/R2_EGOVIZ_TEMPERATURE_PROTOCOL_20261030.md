# EgoViz readout temperature probe — frozen before execution

The existing 81,920-parameter joint checkpoint has development Accuracy 55.70%
but mean confidence 33.35%. Test whether its scores are simply too compressed,
without changing its weights or training another encoder or readout.

Load the existing aggregate, checkpoint and saved predictions only after
checking their hashes. Reproduce selected train metrics and development
probabilities. Use the 332 training labels to fit one positive scalar `alpha`
that minimizes four-choice NLL of `alpha * logits`; choose alpha in [0.05, 20]
by deterministic bisection of the convex NLL derivative. This is a train-only
candidate: evaluate it once on the same 307 development examples, alongside
unscaled logits. No development metric influences alpha. Because the checkpoint
was itself selected on this development set, its result is exploratory, not
an independent audit. In-sample calibration on the training examples may
overfit and is not assumed to improve development calibration.

For a separate diagnostic, assign the 20 development session groups to five
folds using sorted IDs, NumPy default RNG seed 17 and equal-sized folds. Fit
the same one-scalar method on four folds and predict the held-out fold. Every
development session must appear in exactly one holdout; no held-out target is
used to fit its own fold's scalar. This diagnostic fits on development labels,
so it cannot be deployed as a single frozen scalar and is not a blind result.
Record every fold's assignments and scalar, plus all per-example probabilities.

Report Accuracy, NLL, Brier, ECE and mean confidence for baseline, train-only
scalar and out-of-fold diagnostic. Use a paired 20-session bootstrap with 2,000
draws and seed 17 for baseline-minus-candidate NLL and Brier. A positive scalar
must preserve candidate ranking; violations fail closed. Save checkpoint/input
hashes, exact command, environment, and a new output directory. Reload and
verify the saved scalar's predictions. Do not use sealed data, refit the model,
pick a scalar based on development performance, or alter the existing run.

The previous high train/development gap and this small, selected development
set prevent any claim of general four-modality quality. The 95–98% goal remains
open. This test addresses probability scaling only; it does not resolve the
underlying ranking/generalization problem or test long-budget grokking.
