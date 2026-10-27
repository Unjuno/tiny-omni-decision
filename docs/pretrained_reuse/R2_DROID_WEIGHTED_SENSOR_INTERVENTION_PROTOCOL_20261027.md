# Fixed DROID weighted-head sensor interventions

Freeze before computing interventions. Reuse the pinned 80,998 train windows,
the five saved history-plus-sensor heads from change-weighted attempt 02 and
their exact prior OOF probabilities. Verify all dataset/model hashes, recording
folds, training-only normalizers and aligned prediction replay. Do not fit,
alter weights, select checkpoints, access validation/audit data, or use a GPU.

Keep each row's original previous command and target. Compare (1) the original
sensor, (2) standardized sensor coordinates all zero, equivalent to the
outer-training-fold feature mean, and (3) a deterministic within-recording
cyclic donor shifted by floor(recording window count / 2). The shift must be a
permutation with no same-row donor, same recording and same outer fold; all
recordings have at least 24 windows, so input sensor windows do not overlap.
The donor can come from a future position: this is a negative-control input
intervention, never an inference feature or a training example.

For all three conditions report raw weighted-head and fixed algebraic
weight-corrected probabilities, natural-prevalence Accuracy, NLL, Brier, ECE,
mean recording Accuracy, change detection recall/precision and stable false
change rate. Compare aligned to each intervention with paired recording-group
bootstrap, 2,000 resamples, seed 17, percentile 2.5/97.5 intervals for pooled
Accuracy and NLL differences. Sample recordings with replacement and retain
all their windows; report the exact number of groups and rows. Save per-row
probabilities and all hashes outside Git. No tuning, threshold selection or
held-out filtering is permitted.

This is a nonblind train-group control, not independent site/operator or
product-quality evidence. Temporal shifts preserve recording identity but can
create incompatible history/sensor pairs, so performance changes alone do not
prove causal temporal reasoning. The previously observed 95–98% goal remains
open, as do image, audio, video and joint-fusion gates.
