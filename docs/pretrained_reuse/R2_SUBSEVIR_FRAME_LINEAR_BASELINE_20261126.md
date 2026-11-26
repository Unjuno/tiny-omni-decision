# sub-SEVIR one-frame input baseline — 2026-11-26

## Scope and protocol

This is one fixed, task-diagnostic comparison on the already exposed Path A
train/validation split. It did not select or alter the prior LSTM, and no
sealed audit or publisher validation/test data were accessed. It uses only
the last of the eight permitted preceding sensor frames: concatenated frozen
DINOv2-small features for satellite and radar (`[1536]`). A fresh linear
two-class readout was fitted on train windows only. Feature normalization was
fitted on train rows only.

- Seed: 17; device: CPU; no GPU or cloud compute.
- Readout: `Linear(1536, 2)`, 3,074 trainable parameters; full-batch PyTorch
  LBFGS with strong-Wolfe line search, maximum 50 iterations, L2 weight
  coefficient `1e-4`.
- No validation-based checkpoint selection, thresholding, tuning, or repeated
  model search. One deterministic fit was run; the subsequent v2 output is a
  reproduction with saved-weight reload checking, not a second candidate.
- Fit time: 0.480 seconds. Frozen DINOv2-small: 22,056,576 parameters.
- Train: 15,828 windows / 3,969 event IDs. Validation: 3,988 windows / 998
  event IDs. Validation sample-ID-order SHA-256:
  `4d576626463db2cf1ce8ddb7aefffdc4387315b5476ee358dd05cb5625ec834b`.
- The validation was already exposed during Path A model selection. Treat this
  as a development diagnostic, not a blind result.

## Results

| Readout | Train Accuracy | Validation Accuracy | Validation NLL | Validation Brier | Validation ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Last-frame linear | 94.98% | 88.72% | 0.3197 | 0.1763 | 0.0588 | 0.9434 |
| Selected two-stream LSTM, 8 frames | not comparable: last-epoch rolling fit metrics | 89.62% | 0.2380 | 0.1485 | 0.0194 | 0.9103 |

The paired event bootstrap (10,000 replicates, seed 17) for **linear minus
LSTM** gives Accuracy `[-2.18, +0.38]` percentage points, NLL `[+0.0498,
+0.1159]`, and Brier `[+0.0115, +0.0444]`. Accuracy uncertainty includes zero;
the temporal LSTM has better NLL and Brier throughout these intervals. The
linear head's train/validation Accuracy gap is 6.26 points, with NLL rising
from 0.1300 to 0.3197 and ECE from 0.0057 to 0.0588. This small baseline also
overfits and becomes overconfident.

Interpretation: the fixed last-frame representation already achieves close
validation Accuracy, so this exposed cohort does not establish an Accuracy
gain from temporal modeling. The LSTM does improve proper scoring and
calibration relative to this single-frame linear readout. This does not
establish that the dataset is sufficient, that the task is representative, or
that either model meets product quality. The separate 96.49% previous-label
carry-forward figure is an oracle label-history diagnostic and is not an
input-equivalent model baseline.

## Reproduction and artifact integrity

Script added at
[`scripts/evaluate_subsevir_frame_linear_baseline.py`](../../scripts/evaluate_subsevir_frame_linear_baseline.py),
source commit `6ab8336552123845e77ec352c21cb121740a4f4d`, script SHA-256
`65ba2c1f29c094b26175b2d424413e9d27a79dabe39fc64ee54514a536127f41`.
The exact command was:

```powershell
python scripts/evaluate_subsevir_frame_linear_baseline.py `
  --manifest C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-patha-manifest-seed17-v1.json `
  --features C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-dinov2-frame-cache-v1\frame_features_fp32.npy `
  --frame-indices C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-dinov2-frame-cache-v1\frame_indices.npy `
  --feature-metadata C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-dinov2-frame-cache-v1\metadata.json `
  --reference-predictions C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-patha-dinov2-lstm-v1\joint\validation-predictions.jsonl `
  --output C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-frame-linear-baseline-v2
```

Manifest SHA-256: `3adcaf1713a562772ff44ba0bbf15533c411eb187699b96d0764139745e46b4b`.
Frozen feature file SHA-256:
`93b13f70576153c77a019fb1d9583d13d8a419671c6dce1f66973dd465b3c82d`.
The run summary SHA-256 is
`eed35a1fecd839707434b184df1b5efead609351f7189f1ce8730b179b303d77`.
The saved head is 12,432 bytes (SHA-256
`aeffcaf1cea7cc0c43483b4420e49d5fbc76febef7684f643cedb22c6f910dc9`); the
train-only standardization file is 11,441 bytes (SHA-256
`59440d9dcec48daf180c7d52116e44055f5bb50330f1fd8907e79a2e7cab0358`).
The prediction file is 655,510 bytes (SHA-256
`00a651b45aae73f7243324b385e211bdd57ba2a53a98df609514dfe64c82eff7`).
Reloaded saved weights and standardization reproduce the stored probabilities
with maximum absolute difference `0.0`; an independent audit also confirmed
all 3,988 sample IDs, order, and targets match the frozen manifest.

All run artifacts remain outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-frame-linear-baseline-v2`.
No prior corpus, checkpoint, feature cache, or legacy worktree was modified.
