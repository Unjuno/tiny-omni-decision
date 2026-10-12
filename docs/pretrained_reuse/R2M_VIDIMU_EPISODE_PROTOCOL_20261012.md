# Fixed VIDIMU episode-level frozen-feature comparison

## Scope and limitations fixed before validation feature extraction

Compare video-only, sensor-only and joint evidence for 13-way activity decisions
on complete recorded trials. This is one native no-text task toward the Omni
goal, not a replacement for it. It cannot establish frame-synchronized fusion,
online decisions, or arbitrary condition following. Exact alignment remains an
open gate: retain and report the 27 missing and two ambiguous offset records.
Use original complete trials, without applying uncertain offsets or discarding
trials. A missing/mismatched video blocks this fixed comparison rather than
silently shrinking validation. The sealed audit is not loaded.

## Fixed inputs

- Previously frozen subjects: 130 train, 39 development validation, all13 labels.
- Processed original quaternion .sto, all samples including initial pose; eight
  equal-duration mean rotation-entry bins (360 features), with named placement
  retained. This cache already exists for training only.
- Frozen VideoPrism114,365,184 parameters, checkpoint SHA256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  Reuse the verified preprocessing: eight endpoint-spaced frames,288px, with
  actual decoded timestamps recorded. Mean spatial tokens separately within
  each temporal position; flatten eight768-dimensional vectors (6144 features).
- No text encoder, generated captions, inferred label text, filenames or
  participant IDs supplied to the head. Class labels are A01 through A13.
- Standardization fits train only, independently per feature. Constant train
  columns use scale1. Sensor placement is a two-value explicit indicator shared
  by all three conditions, so a video arm cannot be penalized for lacking this
  known acquisition metadata. Report a placement-only prior as a fourth control.
  Report conditional Accuracy/NLL within both placements as well as all13 classes.

## Controlled fit

Use seed17, linear13-logit readouts, no hidden layers, float32 CPU, full-batch
AdamW(lr0.001,weight_decay0.01),100epochs. All three arms use identical examples,
order and schedule. Evaluate epochs0 through100 and select earliest minimum
validation NLL per arm. Preserve final and selected states, normalization, full
history, source hashes, IDs, probabilities and targets. No additional seeds or
hyperparameter search in this initial comparison. Report the trained head count
and full resident encoder count separately; no parameter count excludes towers.

## Evaluation and adoption rule

Save Accuracy/NLL/Brier/ECE10bins, macro class recall, per-class and per-subject
metrics, placement-conditioned metrics, train/validation gap, wall time/RAM/VRAM,
exact save/reload and option-ID permutation checks. Placement prior is Laplace
smoothed from training counts; never use validation labels to fit it.

For the selected joint model, shuffle each modality independently within the
same placement (seed17 fixed derangement), retaining every validation ID; report
which paired labels remain coincidentally equal. Also report missing-modality
stress with standardized zeros, explicitly outside the complete-input training
distribution. Do not claim missingness robustness from this diagnostic.

Compare joint NLL to each unimodal arm with subject-level paired resampling,
seed17/10,000 replicates/percentile95% intervals. Only three validation subjects
means weak uncertainty resolution and checkpoint-selection bias; this is not a
blind test. Consider fusion supported only if joint improves over both arms and
the perturbations support useful contributions from both modalities. Otherwise
retain the negative result. No product promotion or compression follows merely
from success on this narrow task.

Before the full comparison, verify one training video and a separate-process
training-only extraction/readout smoke. This document freezes the intended
experiment; it does not claim training or evaluation has happened.
