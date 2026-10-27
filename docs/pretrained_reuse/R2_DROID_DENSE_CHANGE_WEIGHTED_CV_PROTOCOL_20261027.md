# Dense DROID grouped CV: train-only change weighting

Freeze this follow-up after observing the previous train-fold diagnosis. Use
the exact 80,998 pinned train windows, original five recording-group folds,
feature transformations, two linear heads, seeds, batch size, AdamW settings
and fixed 50-epoch schedule of the preceding experiment. No validation,
audit, video, encoder inference or GPU is used. Retain all windows, all seven
targets and natural held-fold prevalence; do not sample or filter held rows.

For each outer training fold, compute `changed = target != previous_command`
using its training rows only. Set stable row weight `N/(2*N_stable)` and
changed row weight `N/(2*N_changed)`; both groups contribute half of the
training weight and the mean weight is one. Apply weighted seven-class cross
entropy to both history-only and history-plus-sensor heads. Start from zeros,
use the same fold/epoch permutations as the earlier run, train 50 epochs without
held scoring during training or checkpoint selection, and stop on nonfinite
loss/gradients. Save training curves, weights, per-fold sampling weights,
normalizers and all OOF probabilities in a new external directory.

Report raw weighted scores and a fixed, untuned probability correction: for
each option divide its predicted probability by the corresponding training
weight (stable if the option equals the known previous command, changed
otherwise), then renormalize. This is an algebraic reweighting correction,
not a guarantee of calibration. Report natural-prevalence Accuracy, NLL,
macro NLL, Brier and ECE; additionally report changed-command exact accuracy,
binary change-detection recall, precision and stable false-change rate using
the ordinary seven-class argmax. Compare against pinned natural-loss OOF
probabilities on the same rows. Do not tune thresholds or weight ratios on
held outcomes; do not drop hard examples.

This is a hypothesis-driven follow-up on already observed train-fold results,
not a blind evaluation. Windows from the same recording are correlated;
operator/site independence, physical significance of axis changes, video and
multimodal quality remain unverified. Cost-weighted or corrected scores must
not be presented as a final calibrated product model.
