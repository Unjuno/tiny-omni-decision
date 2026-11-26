# R2 fresh AV temporal matching readouts — 2026-10-21

## Question and scope

This development-only experiment asks whether preserving an eight-frame
VideoPrism feature sequence helps a small audio-conditioned decision readout
identify which of two same-video, same-label audio windows aligns with a video
interval. It compares an audio-only scorer, a matched pooled-video scorer, and
an LSTM sequence scorer. It does not evaluate a general AV decision model,
product quality, or any sealed/official test split.

The source is Perception Test official TRAIN only. Prior-exposure media were
excluded by identity before selecting 96 train and 32 validation video groups.
The two groups are disjoint by video; the labels and validation groups are
development data and the validation set was used for checkpoint selection.
It is therefore not blind or independent evidence. No official test or sealed
audit was opened.

## Integrity and frozen inputs

| Item | Value |
|---|---|
| Attempt directory | `C:\\CodexArtifacts\\pretrained-reuse-r2\\perception-av-temporal-holdout-v2-20261021-01` |
| Task | `perception-av-same-video-same-label-temporal-match-v2` |
| Train / validation groups | 96 / 32; 128 distinct videos |
| Manifest SHA-256 | `256cbb0bc30ec183a2e21b1b79d5f3c903b502a95d5e7cfbb5f2956d16bebf74` |
| Prior exposure receipts checked | 1,122 |
| New media | 96 video + 96 audio assets; no within-type duplicates |
| Prior media overlap | 0 video hashes; 0 audio hashes |
| Train/validation video or media overlap | 0 |
| VideoPrism source / weights | `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` / `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` |
| Video feature output SHA-256 | `5ccc0c26d276c7b2683e04c3d4124d963a3abcab21bb69000378a0b0a198f73b` |
| EfficientAT source / weights | `a425fdce92572e602a1d5634799bd9f1f2efa806` / `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` |
| Audio feature output SHA-256 | `ff6d4949efbbdb5ad6a78255a38b32a1ce99aed702fa6977608b89f6c2635bf5` |
| Readout result SHA-256 | `4d454253a011ea431ee01bbe26f72f13e39c1dc9497a3d4a67df4c0cc105a250` |
| Prediction SHA-256 | `9a98b825bcf8f14f628ab6ccd797fe4f1b49792571391c2e111fa6ea56909189` |
| Readout script SHA-256 at final source review | `9491cd5cf2b3286cfafe5043cc61a0576c535c83d1b36fbc1e39ad485333b417` |

Both feature manifests matched the frozen task manifest. The fitting loader
rechecked all feature row keys, array shapes, finite values, and per-array
hashes. The media gate checked every new file, 1,122 prior exposure receipts,
and split identity. These integrity checks passed.

## Fixed fitting protocol

All three heads use seed 17, full-batch AdamW, learning rate 0.002, weight
decay 0.001, 2,048 optimizer updates, and validation every 32 updates. Input
normalization uses train-only feature means and standard deviations (floor
1e-5). The sequence head conditions an LSTM initial state on each audio option,
then attends over the frame states. The pooled control averages the same eight
frame features to one step before using the same architecture and parameter
count. Audio-only scores each option independently. Model sizes are 12,353
parameters (audio-only) and 50,625 (each AV head). The frozen encoders contain
1,961,239 EfficientAT and 114,365,184 VideoPrism parameters and were not
fine-tuned.

The 32-row validation set selects the lowest categorical NLL; ties keep the
earlier step. This is a deliberately small diagnostic and selected-checkpoint
quality is optimistically biased by using these rows for selection. The full
2,048 updates were completed for every arm; best checkpoints and final
checkpoints are both saved outside Git. Each update uses all 96 training
groups, so this is repeated full-batch fitting, not 2,048 new unique examples.

## Results

Selected checkpoint and final step (Accuracy / NLL / Brier / ECE-10):

| Arm | Best step | Selected validation | Final-step validation | Train at 32 | Train at 2,048 |
|---|---:|---|---|---|---|
| Audio-only | 0 | 0.531 / 0.700 / 0.506 / 0.021 | 0.688 / 1.197 / 0.690 / 0.188 | 1.000 / 0.043 | 1.000 / 0.000 |
| Pooled AV | 0 | 0.594 / 0.683 / 0.490 / 0.083 | 0.719 / 1.348 / 0.728 / 0.219 | 1.000 / 0.004 | 1.000 / 0.000 |
| Sequence AV | 0 | 0.625 / 0.683 / 0.490 / 0.113 | 0.719 / 1.305 / 0.715 / 0.207 | 1.000 / 0.004 | 1.000 / 0.000 |

The final-step pair in each cell is validation Accuracy/NLL; the last two
columns are training Accuracy/NLL. At step 128 the audio-only, pooled AV, and
sequence AV validation NLLs were 0.873, 0.979, and 1.123, respectively. Their
best-NLL checkpoints all remained at initialization (step 0). Thus the
sequence model's selected 0.625 accuracy is an untrained random-head result,
not a learned temporal improvement.

The paired sequence-minus-pooled accuracy delta was +0.031 (5,000 row-level
paired bootstrap resamples, seed 17, 95% interval [-0.063, +0.125]);
sequence-minus-audio-only was +0.094 (interval [-0.094, +0.281]). Each
validation row is a unique video group. The interval includes zero in both
comparisons and is coarse with only 32 groups.

Input controls on the selected sequence checkpoint gave 0.625 Accuracy with
time-reversed frames (identical to the original), 0.563 with video zeroed, and
0.563 with audio zeroed. The time-reversal result provides no evidence that
the selected untrained readout uses temporal order. Zeroing either input
changed predictions for some rows but is not a trained counterfactual
complementarity result. The video-only uniform baseline is 0.563 Accuracy,
with NLL 0.693 and Brier 0.500; identical video is presented for both options,
so it cannot distinguish their identity.

## Interpretation and next step

The learning curves show severe overfitting: all heads reach 100% train
accuracy by update 32 while validation NLL worsens, and the fixed NLL selector
retains the initialization. This rules against claiming that simply adding a
sequence LSTM to these frozen features improved this task under the fixed
protocol. It does not establish a general temporal-representation ceiling:
the cohort is small, the task is narrow, and the heads rapidly memorize it.
The sequence-vs-pooled interval is inconclusive, and the time-reversal control
is especially weak because the selected model is untrained.

Do not tune on these validation groups or enlarge this same experiment after
seeing its result. Any next temporal study needs a newly fixed, larger
video-group split and a training contract that can avoid immediate memorization;
it should compare a temporal head against the pooled control on exactly the
same rows and preserve a fresh untouched evaluation cohort. No head is
promoted, no teacher/checkpoint is changed, and this study is not evidence for
the project's 95–98% four-modality objective.

## Reproducibility artifacts

The immutable result directory is
`C:\\CodexArtifacts\\pretrained-reuse-r2\\perception-av-temporal-holdout-v2-20261021-01\\readouts-sequence-20261021-01`.
It contains `effective_config.json`, `training_history.jsonl`,
`validation_predictions.json`, per-arm best and final checkpoints,
`train-normalization.npz`, and `result.json`. The source entry point is
[`fit_perception_temporal_match_sequence_readouts.py`](../../scripts/fit_perception_temporal_match_sequence_readouts.py).
Execution used Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6, CPU for fitting;
the RTX 3080 remained occupied by the unrelated Orbitune process. Fit time was
30.0 seconds. No cloud compute or additional model download was used.

Best/final checkpoint hashes (SHA-256): audio-only
`03aead081cf49f4f2dc57f4e3c6af8150aee8c4a86b5869440df965157e29bc7` /
`9115cf0abd2e88c571cf4016df414024898e32b7dc669ce5837e069e2ea03b02`;
pooled AV `c2e649afb24d1a5ee88e7e0f296420772f2d4e0cdc9eda64aef94fde6fe8f86a` /
`1d23622164c6b7a3f2911bec081962fc5d5b8ce62c322078295d92d135ff1453`;
sequence AV `7122796655fecf75c216364d749b5a3fdaebb247dcaa866bead4f63bf2f2ca66` /
`df6283a59dc0046f5ffde140a88368f56a9510828ba05c368a3c640e5da7d7aa`.
