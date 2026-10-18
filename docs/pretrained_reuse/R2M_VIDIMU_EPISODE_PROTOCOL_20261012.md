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

## Fixed perturbation implementation

`grouped_derangement` hash-sorts unique episode IDs using seed17 separately in
each placement group, then rotates by one. It uses no target labels and is
stable under input ordering. This is a deterministic derangement, not uniform
random sampling over all derangements. Singleton groups and duplicated IDs
fail closed. Six CPU tests verify these properties.

The fixed39-trial validation mapping is saved before feature extraction at
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-validation-control-01/permutation.json`.
It has zero self-pairs. Counting activity labels after fixing the permutation
found zero same-activity replacements; no seed or permutation was retried to
obtain this outcome. Only frozen manifest metadata was read, not validation
observations. Use this exact mapping for each modality perturbation.

## Training-only CPU smoke completed

Source `85c25a0c9b09ea7cac5793755048342c192428a4`, command
`python -m scripts.smoke_vidimu_readout`, separate Python process, Torch2.6.0+cu124
on CPU float32. Two full-training-batch updates, seed17; 4,719 trainable readout
parameters; 1.04 seconds measured body wall time. CE was2.7330 then2.4812 before
each update. Sensor placement is read from actual sensor column metadata, not
the target labels. Exact probability parity after weight reload passed.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-readout-smoke-01` includes
weights, train-fit normalization, training probabilities and result/input hashes.
This disposable smoke is not used to initialize the planned comparison. No
validation/audit data or video encoder was loaded; no model quality, general
fusion, or whole-model parameter-budget conclusion follows from this check.

## Validation sensor preparation after protocol freeze

The identical frozen sensor transform was applied to all39 validation trials
with `python -m scripts.cache_vidimu_training_sensor --split validation`.
No failures or exclusions occurred; shape39x8x5x3x3. Exact float32 feature-array
hash overlap with the pinned130-trial training cache is0. This is an exact
duplication check, not a near-duplicate or physical synchronization guarantee.

Output: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-validation-sensor-features-01`.
Features SHA256: `1329f39d60b179446bcef095f7685e58bba478925b56196ebc877f3f0237486e`.
Records SHA256: `fbaf822aff1ce4844dd6481470b4e84c2fa9130d3f56a21ad29972a2b23a6792`.
The CLI exposes only train/validation; its sealed-audit rejection is regression
tested. Existing training cache and smoke artifacts were not overwritten.
Validation observations have now been read for feature extraction, but no
validation predictions, selection or hyperparameter tuning have occurred.
Sealed audit observations remain unopened. Video download remains in progress.
