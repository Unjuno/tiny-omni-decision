# SoundActions train-only grouped cross-validation — 2026-11-21

## Purpose and status

This follow-up checks whether the audio/video readout comparison from the
48-recording development validation was an artifact of that small selection
set. It uses only the existing 246 labeled SoundActions training recordings.
The original development validation recordings do not enter fold assignment,
checkpoint selection, fitting, or metric calculation. The sealed audit was not
opened, extracted, or evaluated.

This is narrow task evidence, not a new product candidate. The source has one
recordist, the data are small, and the label distribution is imbalanced. The
outer folds estimate performance on held-out recordings from this task; they
do not estimate performance on unseen people or other domains.

## Frozen inputs and protocol

- Source task: SoundActions v2, Zenodo record 23971067; existing source/split
  manifest: `manifests/pretrained_reuse/soundactions-v2-source-20261121.json`
  (SHA-256 `a6473a2fb3f59390d7b9c8d4b93ecd11d1fa2053ca27df7a5c14b5eb10d2c013`).
- Feature caches were read-only. Audio cache SHA-256:
  `ecb678bbfd97c3ad17becfbdd758a926b981a392815b050cbc7b247ba80fede5`;
  video cache SHA-256:
  `012b1b0035eca23607292db899b6ccb18a7f06b55ff3e78b8c670419beff15ce`.
- The compressed NPZ files contain both train and development-validation
  features and were loaded as containers. The script selected only the 246
  `train` rows; it checked that recording IDs are unique and used IDs as the
  explicit group key. No validation row was assigned to a fold or sent to a
  model.
- Class counts in the eligible train set: impulsive 102, sustained 59,
  iterative 85. Each recording was assigned to one of five stratified outer
  folds, seed 17. Every recording received exactly one out-of-fold prediction.
- Within each outer-train portion, 20% of each class was reserved for inner
  stopping. Lowest inner-stop NLL selected an in-memory checkpoint; exact ties
  kept the earlier epoch. Each fit used seed `1017 + fold`, batch size 16,
  AdamW (`lr=0.001`, `weight_decay=0.0001`), 32 hidden units per LSTM direction,
  maximum 60 epochs and patience 12. This fixed schedule was used for all three
  conditions; no hyperparameter search was run.
- Conditions: audio-only (107,331 trainable parameters), video-only (205,635),
  and gated audio/video (325,187). EfficientAT and VideoPrism features remained
  frozen. Outer-fold metrics were computed only after restoring the inner-stop
  checkpoint.
- Paired intervals use 10,000 class-stratified recording resamples, seed 17.
  Since fold models share overlapping training data, these intervals are
  approximate and are not independent-fold confidence intervals.

## Results

Out-of-fold metrics aggregate all 246 train recordings, each predicted once.

| Condition | Accuracy | Macro accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Audio-only | 68.29% | 65.49% | 0.747 | 0.421 | 0.073 | 0.733 |
| Video-only | 40.65% | 35.01% | 1.067 | 0.648 | 0.038 | 0.441 |
| Gated A/V | 67.07% | 63.73% | 0.825 | 0.463 | 0.091 | 0.726 |

Gated A/V minus audio-only Accuracy was -1.26 percentage points (paired 95%
interval -7.32 to +4.88 pp). The interval includes zero. In the positive-means-
candidate-improves convention, gated A/V's NLL improvement was -0.078 (95%
interval -0.168 to +0.013), also including zero. Gated A/V clearly exceeded
video-only on these recordings, but did not improve on audio-only. Its Brier
and ECE were also worse than audio-only.

| Outer fold | Audio Accuracy / NLL | Video Accuracy / NLL | Gated A/V Accuracy / NLL |
|---:|---:|---:|---:|
| 1 (n=50) | 66.00% / 0.854 | 30.00% / 1.108 | 64.00% / 0.932 |
| 2 (n=49) | 65.31% / 0.743 | 36.73% / 1.072 | 73.47% / 0.741 |
| 3 (n=49) | 65.31% / 0.714 | 46.94% / 0.982 | 59.18% / 0.897 |
| 4 (n=49) | 79.59% / 0.585 | 46.94% / 1.083 | 71.43% / 0.716 |
| 5 (n=49) | 65.31% / 0.836 | 42.86% / 1.088 | 67.35% / 0.836 |

Video-only per-class accuracy was 65.69% for impulsive, **1.69% for sustained**,
and 37.65% for iterative. The small VideoPrism readout has no reliable result
for sustained events in this setup. Audio-only per-class accuracy was 78.43%,
47.46%, and 70.59%; gated A/V was 79.41%, 42.37%, and 69.41%, respectively.
The fusion result does not show an across-class improvement.

The selected inner-stop epochs were 5–10 for audio, 1–5 for video, and 6–9 for
gated A/V. This suggests the small video readout reaches its selected point
very early under the fixed schedule; it does not prove that the frozen encoder
has no useful video information or that a different task would fail.

## Reproducibility and resources

- Authoritative output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-04-train-only-groupcv`.
- Status: `COMPLETE_TRAIN_ONLY_NESTED_CV`; output has 17 JSON files / 192,980
  bytes. It contains all 15 fold-condition histories and outer predictions,
  `out-of-fold-predictions.json`, and `summary.json`. No persistent model
  checkpoint was needed; each inner-selected state was held in memory and
  immediately scored on its outer fold.
- Summary SHA-256:
  `d194a3c2f822b48e8c24a0c76bd5dc6103b127ea861ef393782abc2834e9392c`.
  Out-of-fold prediction SHA-256:
  `965510fcdf1ae55be215db46e59802f43f0ea1ed3ce4a5db7e2c1189ce98c14b`.
- Training script SHA-256:
  `8ad3e9f7a29e45b09f5b9c2dbde6ab43efe9534b25f6aaa07734093220b560c3`;
  source-tree HEAD at execution: `f8adf0af1b70bff000911db22c7f604697ec059d`.
- Local runtime: Windows 10 build 26300, Intel Core i7-12700H, Python 3.11.9,
  PyTorch 2.6.0+cu124, NumPy 2.4.6. CPU-only; CUDA remained uninitialized.
  Sum of the 15 measured model-fit wall times was 108.78 seconds. GPU time and
  cloud cost were zero; peak host RAM was not instrumented.
- Command (PowerShell; run in the source checkout):

  ```powershell
  $env:PYTHONPATH = 'C:\CodexWorktrees\pretrained-reuse-next-20261115\src'
  python scripts/cv_soundactions_temporal_readout.py `
    --audio-features 'C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-02\audio-features-01' `
    --video-features 'C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-02\video-features-01' `
    --output 'C:\CodexArtifacts\pretrained-reuse-r2\soundactions-v2-20261121-attempt-04-train-only-groupcv'
  ```

An earlier `attempt-03` run used the same records and produced the same aggregate
metrics, but preceded the explicit group-ID API and its regression tests. It is
retained unchanged. `attempt-04` is the authoritative run; the added group
tests verify that duplicate rows for one recording cannot cross folds and that
conflicting labels for a group fail closed.

## Interpretation and next step

This train-only CV does not support a reliable benefit from adding video to the
SoundActions gated readout. The modest fold-to-fold variation and approximate
paired intervals leave small effects uncertain. Together with the prior
development-validation controls, this is enough to stop tuning this task as a
fusion-quality route. Keep the audio-only result as a task-specific diagnostic;
do not promote it or use the prior validation to calibrate or retune.

The next R2 step should use a different, rights-cleared task where audio and
video carry complementary label evidence, or a new predeclared programmatic
sensor/video Decision task with explicit modality interventions. Do not claim
multimodal quality from feature-cache existence or a favorable synthetic
smoke. The all-modality product goal, 95% per-task target, model package budget,
R0 task-battery gate, and final audit remain open.
