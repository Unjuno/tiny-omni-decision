# LSTM fixed-budget learning curve on dense train folds

Freeze after observing the prior 50-epoch train-fold result and before starting
this longer run. Reuse the exact 80,998 train windows, five recording-group
folds, 880-parameter LSTM, past-only sensor/previous-command inputs, train-only
normalization and change-loss weighting. Use the same fold seeds, batch size,
AdamW settings and epoch permutations as the existing 50-epoch run, and train
from fresh seeded initialization. Do not load prior 50-epoch weights as a
resume: the earlier run did not save optimizer state.

Train each fold for exactly 200 epochs and evaluate its entire held fold only
at epochs 50, 100, 150 and 200, without tuning settings, filtering rows or
choosing a checkpoint from these measurements. Evaluation uses no stochastic
layers and must not change training RNG or optimizer state. At epoch 50 compare
the raw held probabilities against the pinned prior 50-epoch run and fail if
maximum absolute difference exceeds `3e-6`. Preserve all milestones and the
final model in a new output directory; do not overwrite prior results.

For each milestone save natural-prevalence raw and fixed weight-corrected OOF
probabilities, all training losses and model weights. Report Accuracy, macro
NLL, NLL, Brier, ECE-10, per-recording Accuracy, changed-command detection
recall, precision, false-change rate and exact changed-command accuracy.
Compute a paired recording-group bootstrap for epoch 200 minus 50 on NLL and
Accuracy using 2,000 fixed seed-17 resamples. Read no external validation,
audit or test sets. Do not use milestone results to select a product model.

Require at least 1 GiB plus 128 MiB free before starting and 1 GiB plus 8 MiB
before each epoch; stop on nonfinite loss/gradient or low disk and report an
incomplete experiment. Saved model-only milestones do not contain optimizer
or RNG state and are not an exact resumable checkpoint. Execution is CPU-only.
The same train recordings have already informed the experiment, windows are
correlated, and independent site/operator performance is unknown. This is a
learning-curve diagnostic, not blind validation, proof of grokking, a plateau,
multimodal quality or fulfillment of the 95–98% goal.
