# Native audio and four-video correspondence

## Measured result

Frozen audio24 and video_ordered18 heads perform a new controlled four-choice
native correspondence task. Same-speaker/recording-condition candidates remove
speaker/background as a matching cue. Every quartet supplies all four audio
query roles with fixed videos; each candidate is correct exactly once.

| Policy / split | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- |
| random / train | 99.01% | 0.048313 | 0.020842 | 0.027535 | 96.26% |
| random / calibration | 92.46% | 0.222276 | 0.119465 | 0.025372 | 89.97% |
| random / validation | 94.85% | 0.145027 | 0.077345 | 0.023193 | 92.68% |
| hard / train | 94.26% | 0.200168 | 0.102633 | 0.079159 | 86.37% |
| hard / calibration | 80.78% | 0.501151 | 0.276354 | 0.028495 | 79.15% |
| hard / validation | 84.08% | 0.413304 | 0.226742 | 0.018897 | 82.66% |

Calibration-only search chose query temperature1.0 from the fixed17-point
grid on3184 calibration roles; raw and calibrated outputs are therefore equal.
Random94.85% falls to84.08% with near-command negatives. This is a task
difficulty difference, not paired model improvement or a new blind result.
Train/dev gaps are4.15 and10.18 percentage points respectively; the hard task
shows substantially weaker transfer. Do not replace it with the easier score.

## Fixed evidence controls

| Policy / intervention | Original Accuracy | Original NLL | Corrected Accuracy |
| --- | --- | --- | --- |
| random / audio_absent | 25.00% | 1.386294 | N/A |
| random / video_absent | 25.00% | 1.386294 | N/A |
| random / audio_swap | 1.69% | 7.926633 | 94.85% |
| random / video_swap | 1.94% | 7.921553 | 94.85% |
| hard / audio_absent | 25.00% | 1.386294 | N/A |
| hard / video_absent | 25.00% | 1.386294 | N/A |
| hard / audio_swap | 5.26% | 5.630592 | 84.08% |
| hard / video_swap | 5.29% | 5.686673 | 84.08% |

Missing inputs use an explicit uniform25% prototype policy. This comparison
does not prove superiority over unrestricted trained unimodal models. Videos
alone are identical across four balanced query roles, limiting a deterministic
video-only answer to25%. Audio-only lacks the opaque candidate-video mapping.
Content swaps under unchanged IDs invalidate original labels; corrected targets
restore matched metrics. Legitimate option+ID reorder is exactly stable on all
7112 development query roles. The fixed matching function sees posterior
arrays only, never source identities, ground-truth labels or transcript text.
This supports use of both streams in this task, not open-world Omni quality.

## Grouped uncertainty and accounting

| Policy | Accuracy minus uniform | Speaker bootstrap95% interval |
| --- | --- | --- |
| random | 69.85% | [67.72%, 71.57%] |
| hard | 59.08% | [54.97%, 62.92%] |

Whole-speaker paired bootstrap10000/seed17/linear percentile95%; all repeated
quartet roles retained within nine clusters. NLL/Brier/ECE intervals in JSON.
3556 roles per policy derive from889 assets, not3556 independent new sources.
Random889 unique unordered quartets; hard874,3496 unique query contexts and60
repeated contexts. All889 parents retained. No train/cal/dev media/source
overlap; hard distance refers to six-slot Hamming distance, not a phonetic
LibriSpeech benchmark. Original51 semantic options and25 letters unchanged.
4080 training assets/398 calibration assets remain; no new training updates.

## Reproducibility, resources and limits

Primary01 succeeded; replay01 failed at integer-key versus serialized string-key
report comparison. All attempts and original execution source remain intact.
Version2 only fixes finite JSON-domain equality, with no rounding/tolerance.
Primary02 and separate replay02 complete all exact prediction/catalog/JSON,
full model reload and terminal hash checks. No performance-driven change.

| Execution | Seconds | Peak RAM GiB | Output MB | Receipt SHA256 |
| --- | --- | --- | --- | --- |
| lombard-correspondence-02 | 26.236 | 1.631 | 5.961 | `9638d06b7fe4c0e3c53e0a0080d38b350eb135fd8cf819e2e3fcfdbf8e977516` |
| lombard-correspondence-replay-02 | 26.648 | 1.629 | 5.961 | `1a75a4aff90f754ba5633c49765e2d858da56ea401218bcebefb564df5dc4047` |

Execution source:`401e8a2b922daaf2d40a710ef054ccc36c7bf8b9`. Exact commands/timestamps/runtime,
all split/checkpoint/catalog hashes, full raw/calibrated/control metrics and
[small receipt](R2_LOMBARD_CORRESPONDENCE_RESULTS_20261015.json) are retained. CPU i7-12700H,4threads/interOp1,
FP32 neural/FP64 scoring, deterministic; zero GPU/cloud/encoder loads,
gradients or model updates. Existing runs/configs/checkpoints remain unchanged.
Composition122790899 registered parameters/coefficients includes native encoders,
two full106355 heads, both2304 normalizers,12 native temperatures and1 query
temperature. No pruning credit; AV slice only, not complete four-modality size.

This reuses selected observed development data and structured spoken commands.
No sealed audit, general-purpose temporal/causal reasoning, sensor fusion,
per-modality95–98%, complete resident package/latency or product adoption claim.
Keep this as an actual evidence-dependent AV baseline; next requires independent
task/OOD coverage rather than tuning these negatives/temperatures/checkpoints.
