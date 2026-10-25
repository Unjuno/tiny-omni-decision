# Path A audio CPU/reference parity replay — 2026-10-26

## Purpose and scope

This is a runtime reproducibility diagnostic for the existing frozen Audio
Path A readout. It investigates the Audio probability-parity failure recorded
in [the typed dispatcher report](PATH_A_TYPED_JSON_DISPATCHER_20261022.md).
It is not a new model-quality benchmark, a checkpoint-selection result, or a
training run. No GPU, sealed audit, or training was used. Existing model,
feature-cache, validation, and reference-prediction artifacts were read-only.

## Fixed replay

- Source commit: `2a882ea7a5f9bc71bdfbd42afadaca48143ad2fd`.
- Dataset attempt: `speech-commands-sample-seed17-128-64-retry2`.
- Fixed validation clips: 640; four query types per clip; 2,560 predictions.
- Runtime: CPU, batch size 8; model load 0.353 s; inference 100.455 s.
- Feature cache SHA-256: `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`.
- Reference predictions SHA-256: `e203cb22ba5e18d75b0a4a78afcd475993f556edefedc7cf12aeb5210a5b3932`.
- Readout SHA-256: `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.
- Ordered media-ID hash: `20dca814ebf78e19ce0d81d77a2f87fec615d9496c20b7567347347ec9073b70`.

## Results

All 2,560 predicted classes match the saved reference. Accuracy is 97.8516%
for both. Probability vectors are not numerically identical: maximum absolute
probability delta is 0.057061, while the median is 0.0000380. Aggregate
metrics are:

| Metric | Saved reference | CPU replay |
| --- | ---: | ---: |
| Accuracy | 0.978516 | 0.978516 |
| NLL | 0.069327 | 0.069768 |
| Brier | 0.032318 | 0.032366 |
| ECE-15 | 0.006127 | 0.006372 |
| Mean confidence | 0.977192 | 0.977083 |

Per-query-type accuracy also matches exactly: direction/presence 97.9688%,
keyword identity 95.9375%, response/presence 98.1250%, and switch/presence
99.3750%. No prediction-class mismatches were found in any stratum.

The existing strict saved-output probability parity threshold of `1e-5` is
still not met. Identical CPU single-item and batch-8 outputs on the prior
four-query diagnostic ruled out batch composition for that diagnostic sample.
An earlier full-cache verification of the same readout on the same 640 fixed
validation IDs, using the persisted encoder features, found exact prediction
indices and only `2.2054e-6` maximum probability delta versus the saved GPU
run. Its source report is
`C:\CodexArtifacts\pretrained-reuse-r2\audio-observation-cache-multiquery-v4\verification-report.json`;
the source run report SHA-256 is
`833258cbc584cbe5f2583c7cc9a5a6f115deb9a46b232e81b3eb7c7489fc081b`.
This makes the saved readout computation an unlikely explanation for the much
larger end-to-end CPU delta. The changed feature-extraction path is the leading
area to investigate, but this replay does not separate Whisper audio features
from MiniLM text features.

Separate one-sample feature probes showed small differences between cached
CUDA BF16-derived features and CPU FP32/BF16 feature extraction. Together
these observations make encoder precision/backend differences a plausible
cause, but do not establish the cause of the 0.0571 maximum output delta over
the full cohort. In particular, the present replay does not establish that
all cached features were generated under identical kernels or deterministic
settings. The largest delta is a `keyword_identity` query on sample
`speechcmd-validation-09a8d685789d8c537ba1`; class and target remain the same,
but the top-option probability changes from 0.6096 to 0.5525. No decision
changes, but the difference is material for probability parity.

No parity tolerance was loosened and no runtime code was changed. Treat the
CPU and saved reference paths as decision-equivalent on these 2,560 fixed
queries, but not numerically equivalent under the current gate. Preserve the
strict gate until a same-backend feature/reference replay or a documented
precision-specific tolerance policy explains the observed maximum difference.

## Artifacts and limits

Full run metadata and per-example predictions are preserved outside Git at:

`C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-full-validation-cpu-parity-20261026-attempt-01\`

Files: `result.json` and `predictions.jsonl`. The result records `no_gpu_or_training=true`.
This one fixed validation sample does not support general claims about Audio
quality, other hardware, or the four-modality product target.
