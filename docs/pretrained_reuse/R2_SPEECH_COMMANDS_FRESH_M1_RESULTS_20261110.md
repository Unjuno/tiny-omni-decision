# R2 fresh-speaker Speech Commands M1 — 2026-11-10

## Scope and decision

This is a single, preplanned comparison on a new, speaker-disjoint development
cohort. It addresses a narrow question: whether the frozen Whisper Tiny audio
encoder plus a 3,850-parameter task readout transfers to ten fixed English
keywords on previously unused speakers. It does **not** test general speech
understanding, open-vocabulary decisions, JSON decision quality, multimodal
fusion, or the 200M-parameter product target. It is not a sealed audit and does
not establish that the overall project goal is achieved or impossible.

The old baseline was preserved unchanged and was compared once against a new
seed-17 readout. The new readout was trained for eight fixed epochs; the last
epoch was used, without selecting a checkpoint on validation. Temperature was
fit only on a speaker-disjoint 20% calibration subset drawn from training.
Validation did not affect training, temperature fitting, or checkpoint choice.
No sealed-audit data was loaded.

## Frozen cohort and integrity

- Dataset: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`,
  official TRAIN shards only; target vocabulary: `yes,no,up,down,left,right,on,off,stop,go`.
- Train: 1,500 examples / 403 speakers; fit: 1,210 / 322 speakers;
  calibration: 290 / 81 speakers.
- Development: 1,000 examples / 200 speakers, 100 per class.
- Historical and previously observed speakers, source rows, and paths were
  excluded. Train/development overlap by speaker, source row, and source path
  was zero. Cross-split exact audio hash overlap was zero.
- One exact duplicate waveform pair exists **within train** (1,499 distinct
  hashes among 1,500 rows); it was retained and is reported, not silently
  removed. There were 1,000 unique development audio hashes.
- Validation ID-order SHA-256:
  `82d9a30e1edf8ab9fe5e6b2875ca09b50477d375a45a5a8580344967e8f0d1ba`.
- Train manifest SHA-256:
  `dafba4f510d1ca49fe89d58ac9a4ea0f4268081b8689cbafeb25b46739c2df6e`.
- Validation manifest SHA-256:
  `e19e5c44b6c3fc01d247bcd163dfbe1f46b6e95bc59ccbb6d800f90de3ea3ec9`.

## Model and execution

- Frozen encoder: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`,
  37,760,640 parameters, zero trainable; checkpoint SHA-256
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Readout: 3,850 trainable parameters, seed 17, 8 epochs, batch 64,
  AdamW (`lr=1e-3`, `weight_decay=1e-4`), fixed last-epoch checkpoint.
- Train-only calibration: scalar temperature `0.47748440347754983`; held-out
  calibration speakers had zero overlap with fit speakers or development.
- Candidate weights SHA-256:
  `cf5f5b2e552c3adc82afe86650d1c765f5fb2d3e6f3127b1406ba35c3b5ebfa8`.
- Feature cache SHA-256:
  `5db261fb2fbcee07102623f3c43d66fc7bc2c369d47678345525f6aba3c281a8`.
- Frozen feature extraction used local RTX 3080 Laptop 16GB, CUDA 12.4,
  PyTorch 2.6.0+cu124, Transformers 5.6.2, Python 3.11.9, BF16 inference;
  peak allocated VRAM was 327,900,160 bytes. Load took 0.20s, train feature
  extraction 10.64s, development extraction 6.59s.
- Readout fit was CPU-only: 0.173s, 152 optimizer updates; no GPU training.
  Paired evaluation and 10,000-repeat bootstrap were CPU-only and completed in
  about 3.82s. Paid compute: $0.
- Output directories are outside Git under
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-*`.
  Baseline weights/report and prior experiments were read-only.

## Same-cohort results

All metrics use the same 1,000 held-out speaker examples and pinned frozen
features. Accuracy intervals use a paired speaker-cluster bootstrap (10,000
resamples, seed 17, 200 speakers, percentile 95% intervals). Negative paired
metric deltas favor the candidate.

| Model | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Frozen prior readout, fixed prior temperature | 0.947 | 0.1833 | 0.0740 | 0.0149 | 0.9469 |
| Fresh-speaker readout, train-only temperature | 0.951 | 0.1655 | 0.0654 | 0.0225 | 0.9350 |
| Candidate minus baseline | +0.0040 | -0.0178 | -0.0085 | +0.0076 | — |

Paired candidate-minus-baseline 95% intervals:

| Metric | 95% interval |
|---|---:|
| Accuracy | [-0.0039, +0.0118] |
| NLL | [-0.0338, -0.0033] |
| Brier | [-0.0166, -0.0012] |
| ECE-15 | [-0.0038, +0.0159] |

The point accuracy gain is small and its interval includes zero. NLL and Brier
paired intervals favor the candidate; ECE is mixed and the point estimate is
worse. This does not justify further tuning on this development cohort.
Per-keyword recall ranges from 0.92 to 0.98 for the candidate (lowest: `up`,
0.92; `yes` and `off`, 0.98). Both methods score 0.10 against uniform
10-choice guessing; the candidate's training accuracy is 0.917, not a
development-to-training generalization guarantee.

## Artifacts and reproducibility

External run locations:

- frozen media/manifests:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-media-20261110-attempt-04`
- features:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-features-20261110-attempt-01`
- candidate training:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-training-20261110-attempt-01`
- paired evaluation:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-evaluation-20261110-attempt-01`

The paired evaluation stores per-ID target and both probability vectors in
`paired-validation-predictions.jsonl`; SHA-256
`c95344d7a040e6230ce4bdf007e89b51f0ce32c8092e750352aba456bd4d9de0`.
The `comparison-report.json` digest is stored in the adjacent
`comparison-report.sha256` sidecar. Training report SHA-256:
`e45541674b852923851e0305b6cefcbe4957c2de67e028b09afe0160d8da3ae7`.

## Interpretation and next gate

This is evidence that a compact learned readout can retain high accuracy on a
small, closed-set audio keyword task with new speakers. It does not support
95–98% quality across text, image, video, and audio, does not validate temporal
fusion or sensor inputs, and does not show the readout replaces a general
Decision model. It also says nothing about ternary runtime accuracy or the
full product parameter/storage budget.

Do not run more seeds or tune against these development examples. Keep this
result as a narrow R2 building block. The broad goal remains open: define and
admit separate representative tasks for every required modality, establish
independent validation/audit gates, and compare parameter/storage/quality on
those tasks before making product claims. No Teacher v1/v2 artifact, sealed
audit, ternary checkpoint, or existing run was modified.
