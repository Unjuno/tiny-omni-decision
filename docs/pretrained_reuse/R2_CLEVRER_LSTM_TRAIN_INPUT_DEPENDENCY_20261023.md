# CLEVRER temporal LSTM input-dependency audit — 2026-10-23

## Scope

This read-only CPU inference audit asks whether the previously selected
temporal joint LSTM responds to its audio and video inputs on the frozen
official-training split. It does not fit, select, calibrate, or promote a
checkpoint. Validation labels were not forwarded or scored; the sealed audit
was not opened. The checkpoint had previously been selected on the same
task-specific development validation described in the October 22 report, so
this TRAIN analysis is not independent quality evidence.

## Frozen inputs and execution

- CLEVRER official-train synthetic AV v2; 3,696 rows, 773 scenes, 1,848
  opposite-target event pairs. Every donor row belongs to a different scene
  and keeps the same target index. Scene and pair metadata alone determine
  the fixed cyclic donor mapping.
- Exact feature-cache SHA-256:
  `0b146fa637049d84ad88f0a71638cf30f6e967bd66add13ead0ab062bcbebdff`.
- Train JSONL SHA-256:
  `3b323bab2ab52b54bd3d4aa8a14b7bc929c000f440394a8e724a2a88eaa8e9b3`.
- Split and feature-receipt SHA-256:
  `7504eb1add69594d8d25df6b32b280e840b31018a5bd5f6e01baf0d01b3a4081` /
  `d73eb35fa67c031102c20fa9f90718a220b1a9374bcf087468f67a30e0420645`.
- Read-only selected checkpoints: joint LSTM at step 1,792; audio-only LSTM at
  step 1,792; video-only LSTM at step 1,664. Each readout has 48,868 parameters.
  Joint checkpoint SHA-256 `edffaed3711c745db37205afc74ba19c42dcdadbcbf3b8c8dfc318351442195c`;
  audio-only `028684664e964a8cb3316e9d92c4b6168b04c54d6f05038942f0c394cc83ca1f`;
  video-only `8e245cf8c03b2b2314894e364ae31cdc2c9e8fe9e261d192e271c360f4ecd4d8`.
- Frozen encoder revisions are VideoPrism-B `be719a406d563b66f0ac969e7c94bab8e997c81a`
  and MiniLM-L3 `4ca70771034acceecb2e72475f72050fcdde4ddc`. Combined with the
  48,868-parameter readout, this is 131,803,876 referenced parameters for this
  AV configuration. It contains no natural-audio encoder and is not a complete
  Omni model or package-size result.
- Source commit `61b02dbbb84102f87bcaaaec6ecb54ff935a7705`; runner SHA-256
  `c5f92942354d41ccfa68f9823305e410c519f26c607f3c6ab4546f6dea8244fb`.
- CPU-only, Intel i7-12700H, Python 3.11.9, PyTorch 2.6.0+cu124, two PyTorch
  threads, FP32; no GPU, cloud, optimizer updates or cost. Inference took
  2.81 seconds. Peak RAM was not instrumented.

The feature NPZ container includes the historical validation arrays. The
loader validates their existence and split boundary, but only the prefix rows
listed as TRAIN are passed to the model. No validation examples were forwarded
or scored in this audit. The original cache, JSONL files, run metadata and all
three checkpoints were hashed before and after; all nine protected hashes
matched.

## TRAIN results

Metrics below are on the 3,696 rows used to fit the historical readouts. The
perturbations keep the original labels, so they measure sensitivity under
deliberately mismatched inputs; they are not alternative valid benchmark sets.

| Model / condition | Accuracy | NLL | Brier | ECE-15 | Choices changed vs. joint baseline |
|---|---:|---:|---:|---:|---:|
| Joint LSTM, original | 87.26% | 0.2671 | 0.1735 | 0.0107 | — |
| Audio-only LSTM, original | 75.62% | 0.4380 | 0.3051 | 0.0092 | — |
| Video-only LSTM, original | 50.00% | 0.6931 | 0.5000 | 0.0007 | — |
| Joint, zero audio | 50.00% | 0.7840 | 0.5659 | 0.1435 | 51.06% |
| Joint, zero video | 66.29% | 0.6459 | 0.4436 | 0.1730 | 30.33% |
| Joint, swap audio across scenes | 68.99% | 0.7416 | 0.4461 | 0.1790 | 33.85% |
| Joint, swap video across scenes | 68.83% | 0.7349 | 0.4453 | 0.1812 | 30.28% |
| Joint, reverse audio order | 33.69% | 2.5986 | 1.1206 | 0.5231 | 68.61% |
| Joint, reverse video order | 71.24% | 0.8527 | 0.4630 | 0.2068 | 30.14% |

These results show that the saved LSTM’s decisions respond to both streams and
to their temporal order on its TRAIN data. Relative to the joint baseline,
zeroing video lowers TRAIN Accuracy by 20.97 points and a different-scene
video lowers it by 18.43 points. Zeroing audio lowers it by 37.26 points.
The audio-only model still reaches 75.62%, and video-only remains at chance,
which is consistent with a strong audio shortcut plus learned video context.
The measurements do not establish how much the video effects generalize.

## Limits and next step

CLEVRER contains no audio; this task uses a generated 880-Hz tone. Therefore the
result demonstrates only that this frozen-feature LSTM learned to use both
synthetic temporal streams on its training examples. It does not establish
natural AV, audio/video complementarity on unseen sources, or the project's
95–98% quality target. The earlier 83.51% validation result remains the old,
exposed development estimate and was not re-evaluated here.

This helps locate the previous conditional-readout failure: its near-chance
scores cannot be generalized to the temporal LSTM, which learns a stream- and
order-sensitive mapping on the same training corpus. It does not isolate the
cause to architecture alone; training device/backend, candidate representation,
and readout capacity also differ. Next, freeze a permitted task in which video
content is indispensable and audio-only shortcuts are balanced, then compare
temporal and conditional readouts on new grouped development IDs. Do not use
these TRAIN perturbations to select a product checkpoint.

## Artifacts and verification

Command:

```powershell
$env:PYTHONPATH='src;.'
python -m scripts.audit_clevrer_train_input_dependency `
  --config C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-temporal-decision-v2-coverage-attempt-01/effective-config.json `
  --feature-dir C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-coverage-v2-decision-features-attempt-01 `
  --feature-report C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-coverage-v2-decision-features-attempt-01/feature-report.json `
  --sample-dir C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-coverage-v2-samples-attempt-01 `
  --run-dir C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-temporal-decision-v2-coverage-attempt-01 `
  --output-dir C:/CodexArtifacts/pretrained-reuse-r2/clevrer-lstm-train-input-dependency-20261023-01
```

Output `results.json` SHA-256:
`7a5beee635bd7fffa55e24e9a99e58e6ed92d94293e544eceb7fa23bf99cb5ad`;
`run-metadata.json` SHA-256:
`f541dec273267426d9f8d593a7cf8673aa3020757387e1abb9ee196c607fca4b`.
All nine per-row prediction files are in the new artifact directory; their
hashes are recorded in run metadata. Targeted tests: **3 passed**. Ruff passed
for the runner and tests. The full repository suite and GitHub Actions are
checked after publication.
