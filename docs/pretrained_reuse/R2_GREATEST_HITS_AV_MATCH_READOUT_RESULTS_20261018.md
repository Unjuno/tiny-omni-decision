# Greatest Hits AV match readout results (2026-10-18)

## Scope

This is a narrow binary correspondence probe: given an audio recording and a
video recording, predict whether they came from the same recorded material
event, returning probabilities in `[mismatch, match]` order. It is not a test of
general multimodal reasoning, temporal synchronization accuracy, or the
product's four-modality objective. No official validation/test partition or
sealed audit was opened.

The frozen corpus contains 252 train rows / 63 pair clusters / 126 audio and
126 video source groups; 76 calibration rows / 19 clusters / 38 groups; and 112
validation rows / 28 clusters / 56 groups. Validation is low-power. Every
cluster has four rows and balanced positive/negative targets for each individual
audio and video source, so an audio-only or video-only model should be at chance.

## Fixed comparison

Source commit: `42b6c3f0eec0371d7e20c7dfbe080898651a6ee5`.
Seed 17, AdamW at 0.001, weight decay 0.01, 100 epochs, no early stopping;
one four-row cluster per update. Each arm presented 25,200 row instances over
6,300 updates, with 252 unique training rows and 63 unique pair clusters.
Checkpoint selection was the earliest minimum uncalibrated validation NLL over
epochs 0–100. A single positive temperature was fit on the separate calibration
split after checkpoint selection. The readout arms were fresh audio linear,
fresh video linear, and paired 64-dimensional audio/video projections with a
cosine score. Encoders stayed frozen.

| Arm | Raw parameters before quantization | Selected epoch | Validation accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio only | 1,962,009 | 40 | 50.00% | 0.693155 | 0.500008 | 0.00162 | 0.50162 |
| Video only | 114,366,722 | 36 | 50.00% | 0.693148* | 0.500001* | 0.00043* | 0.50043* |
| Joint AV | 116,400,151 | 1 | 53.57% | 0.693052* | 0.499905* | 0.03140* | 0.50431* |

Starred rows show validation metrics after temperature fitting on calibration;
all other table values are the selected checkpoint before calibration. The
video and joint temperatures reached the allowed upper bound of 20.0, which
flattens probabilities and should not be interpreted as robust calibration.
Joint AV's uncalibrated selected validation metrics were Accuracy 53.57%, NLL
0.714005, Brier 0.518461, ECE-15 0.10122, mean confidence 0.58334.

At epoch 100, joint AV reached 100% train accuracy / NLL 0.0000044 but validation
was 55.36% / NLL 2.61451 / Brier 0.76812 / ECE-15 0.39140 / mean confidence
0.93797. This is a pronounced train/validation overfit signal. The selected
joint checkpoint was epoch 1, not the final weights. Audio and video-only
validation accuracy remained 50% throughout, as expected from the balanced
marginals.

Paired bootstrap used 10,000 validation pair-cluster resamples, seed 17, and
percentile 95% intervals. For calibrated joint-minus-audio/video-only, the
accuracy delta was +3.57 points with interval `[0.00, +8.04]` points; NLL delta
was `-0.000103` with interval `[-0.000226, +0.000009]`; Brier delta was
`-0.000104` with interval `[-0.000226, +0.000008]`. These intervals include no
improvement. With only 28 independent validation clusters, the result is
inconclusive and the tiny point-estimate gains are not evidence of useful joint
quality.

## Integrity and runtime record

The training run completed on CPU in 29.04 seconds; PyTorch `2.6.0+cu124`,
Python 3.11.9, four CPU threads, Windows x64; GPU use and cloud cost were zero.
Both pinned feature caches passed fresh-process replay over all 220 groups with
maximum absolute difference 0.0. The audio replay receipt is in
`C:\CodexArtifacts\pretrained-reuse-r2\gh-av-match-audio-feature-replay-01`;
the video replay receipt is in
`C:\CodexArtifacts\pretrained-reuse-r2\gh-av-match-video-feature-replay-01`.

The first execution attempt stopped before any model updates because the
config had a JSON trailing comma. A second attempt completed the audio-only
100-epoch arm, then exposed an inference-tensor/autograd issue during
temperature fitting. That partial output is retained at
`C:\CodexArtifacts\pretrained-reuse-r2\gh-av-match-readouts-01`; it was not
resumed or overwritten. A tested fix cloned calibration logits, and the final
fixed run completed all three arms in the new directory
`C:\CodexArtifacts\pretrained-reuse-r2\gh-av-match-readouts-02`.

Final run metadata:

- Started: `2026-10-18T22:53:24+09:00`; finished: `2026-10-18T22:53:53+09:00`.
- Effective-config SHA-256: `519d7661e2788f6e3573288cc99806a48d9e51fa9eb31312ee4e99f66b4d303b`.
- Result SHA-256: `0df3f087b70856d79db53fe9d3ca1199bfc3578fe51b5b7bd696bda902e807e`.
- Ordered validation predictions SHA-256:
  `2b41e4eca3d0cab6546111b2baea42d20648a977d6fe2c9d4dee2d45af74a442`.
- History: 303 epoch records; each arm has 6,300 updates and separately saved
  best/final checkpoints. Checkpoint digests are in `result.json`.
- All artifacts are outside Git. No existing model, corpus, sealed-audit data,
  Teacher, or Orbitune process was changed; no GPU job was launched.

## Decision

This comparison does not support product adoption or a general 95–98% claim.
It does show that a small learned cross-modal readout can memorize the balanced
training pairs while failing to generalize on the small held-out split. The next
useful step is to improve the task's independent cluster coverage and evaluate
whether the audio/video representations contain transferable event identity;
do not increase model capacity or sweep seeds on these same validation IDs.
This task is only one AV input relation and provides no evidence for text,
image, sensor, or broad video decision quality.
