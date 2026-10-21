# Perception Test AV temporal train-only grouped OOF diagnostic — 2026-10-22

## Question and scope

The prior temporal-scale experiment's 64-video validation cohort has already
been inspected and used for checkpoint selection. This follow-up asks whether a
fixed short training budget transfers to unseen training video groups, and
whether a source-label-only prior explains part of the task's option-position
signal. It does not reuse the exposed validation cohort for tuning or scoring.

This is a narrow train-only diagnostic on the existing frozen task. It does not
measure general audiovisual reasoning or product quality. No model checkpoint
was promoted, and no GPU, official test, or sealed audit was used.

## Frozen inputs and execution

- Source commit: 31f61b8d86bd26e448d280a2607332b41642adf6.
- Manifest SHA-256:
  9147e598a4ac9e8412204b472ac6b32a019517ff682b8587379f23605672dbef.
- Canonical train rows SHA-256:
  55ce47616df071405152b7c45fdc488b73f717dcd8b922e7582fcae352a24e80.
- Video train-sidecar/feature inventory SHA-256:
  4bb6799c97fa7dc39793b8edc50e658e790bcbe7a91acd7660557bf9eae191f0.
- Audio train-sidecar/feature inventory SHA-256:
  d011c79972215a964e4e1876eb7f075cdef4bd2bab8dbb4a66633e3e0ced1e3a.
- Feature file SHA-256, shape, finiteness, manifest sample identity, and video
  identity were checked. The loader reads feature sidecars only under the train
  directories and accesses only the manifest's train rows. The manifest file
  itself also contains validation metadata, but validation rows and validation
  features were not used or scored.
- Runner: scripts/audit_perception_av_temporal_train_oof.py,
  SHA-256
  ddd5ecde8ff9ec39469e4954f093da60c5685c1b2d60ace56b6d002a95a140d5.
- Command:

  ~~~powershell
  python -m scripts.audit_perception_av_temporal_train_oof --manifest C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\task_manifest.json --video-train-dir C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\features_videoprism_per_frame_mean\train --audio-train-dir C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\features_efficientat\train --output C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\train-only-oof-20261022-02
  ~~~

- Run metadata SHA-256:
  fb39818c0bf417fe192150909690fd97ef329e334a912a5f89d4c630c140bbab.
- Full OOF result, including per-video scores and fold metrics:
  C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\train-only-oof-20261022-02\oof_results.json,
  SHA-256
  b7cb4de737b2737e2b12c96f7ad388fd9987fc67d3ee96c43d4a4226eabc190d.
- Host observation: Windows, Intel i7-12700H, Python 3.11.9,
  PyTorch 2.6.0+cu124, NumPy 2.4.6. The run used CPU only and took 5.42
  seconds. No cloud compute or external cost.

## Fixed protocol

- 224 unique official-TRAIN video groups; deterministic five-fold assignment
  using seed 17, stratified by target position and source event label.
- Every video appears in exactly one held-out fold. Each of the three arms
  trains on the other four folds and produces one out-of-fold prediction for
  every video.
- Audio-only, pooled AV, and eight-frame sequence-AV readouts use the existing
  model definitions, AdamW, learning rate 0.002, weight decay 0.001, and
  exactly 32 full-batch optimizer updates per fold. Normalization is fit on
  each fold's training groups only.
- No early stopping, validation selection, seed search, hyperparameter search,
  architecture modification, or checkpoint selection. Initialization is
  deterministically reset to seed 17 for each fold/arm.
- Total: 480 optimizer updates, 86,016 repeated training group-visits, and
  224 unique held-out groups per arm. Repeated visits are not new data.
- The train-only source-label baseline predicts the most frequent option
  position within each source label using only the other four folds. This is a
  diagnostic baseline, not a model input.
- 95% intervals use 5,000 paired bootstrap resamples of the 224 video groups
  with seed 17. They are descriptive; folds are not independent replications.

## Results

The all-train majority option-position baseline is 56.70% Accuracy. A
source-label-only out-of-fold prior reaches 58.04%, indicating a modest
association between event category and which candidate occupies the target
position. This is not evidence of media leakage; it is a task-construction
shortcut that a model could exploit if trained on it.

| Fixed 32-update OOF arm | Accuracy (95% group bootstrap CI) | NLL (95% CI) | Brier | ECE-10 |
|---|---:|---:|---:|---:|
| Audio-only | 45.54% [39.29%, 52.23%] | 1.2472 [1.0862, 1.4109] | 0.7790 | 0.3314 |
| Pooled AV | 47.32% [41.07%, 54.02%] | 1.6222 [1.3831, 1.8718] | 0.8326 | 0.3779 |
| Sequence AV | 47.32% [41.07%, 54.02%] | 1.5326 [1.3136, 1.7593] | 0.8218 | 0.3421 |
| Source-label-only, fold-local mode | 58.04% | not probabilistic | — | — |

The fixed trained readouts do not transfer above the simple option-position or
source-label baselines. Their confidence intervals overlap one another and
include chance-level performance. High NLL and ECE indicate substantial
overconfidence out of fold. Sequence AV did not improve over pooling on this
fixed protocol.

The label-position table also shows the source label is correlated with target
index: Interaction:hitting has target positions 0/1 in 86/53 groups, while
Interaction:rolling has 14/16. The overall target counts are 127/97. Source
labels were excluded from model inputs.

## Interpretation and next step

This provides additional evidence that the current small Perception Test
readouts do not generalize reliably across video groups under the fixed
32-step budget, despite fitting the full training cohort rapidly in the prior
experiment. It also exposes a mild event-label/option-position prior. The
result does not identify whether the main cause is the task, negative sampling,
pretrained features, the small number of unique groups, or readout overfitting.
It does not establish a ceiling for AV models or recurrent architectures.

Do not extend this fit, increase capacity, or tune against the already exposed
64-video validation cohort. A next AV quality experiment needs a distinct
task/readout plus a newly frozen, sufficiently large video-group validation
cohort and balanced candidate-order construction. Preserve the old validation
results strictly as historical development evidence.

## Tests and artifact preservation

Unit tests cover deterministic group assignment, train-only feature loading,
and fixed-seed grouped bootstrap. Targeted tests: 3 passed. Ruff passed for the
runner and tests. The original manifest, features, prior readout outputs, and
model checkpoints were not modified; the new OOF output is in its own external
directory.
