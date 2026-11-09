# Fresh-speaker Speech Commands transfer results (2026-11-10)

Status: completed as a fixed-checkpoint development transfer measurement. The
provisional audio keyword gate is **not passed**. This is not a product Teacher
promotion, an official Speech Commands test result, a general speech claim, or
a multimodal result.

## Cohort integrity and correction history

The metadata-only preflight selected 1,000 records (100 per class) from pinned
official TRAIN shards, grouped into 562 speakers, with no more than five rows
per speaker. It excluded the prior 1,920-record train/validation exposure by
source row, speaker hash, and source-path hash. The pinned source is
`google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`, v0.02, whose
card identifies CC BY 4.0. The task is ten fixed English keywords only.

The initial selector assigned numeric IDs using alphabetical class order.
That was incorrect: the pinned v0.02 class order is `yes, no, up, down, left,
right, on, off, stop, go`. One direct example was Parquet row
`train-00005-of-00006.parquet:7703` with numeric label `0` and source path
`yes/d8c314c0_nohash_0.wav`, initially called `down`. The initial attempt's
saved outputs are retained under
`C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-01-results`;
`evaluation-validity-audit.json` marks them invalid for quality claims.

The corrected manifest retains the exact same 1,000 sample IDs and ordering,
source rows, audio files, and speaker groups. A metadata-only source recheck
validated every numeric ID against both the source path class and the pinned
dataset card. The corrected manifest has SHA-256
`06baea7a101ce60849b5a6dc81d4413c531c0871091fa914d169a53e5f30f23e`; the
corrected inventory has SHA-256
`83dd38c4247666fb4b3d3a18e22da45cd9e8ab3c3343296a53ae3160be10c816`.

One corrected-label execution completed inference and bootstrap but failed
while assembling its report because the code expected the old materialization
report filename. It saved no predictions or metrics and displayed no score.
The same frozen candidate and cohort were then rerun with only report-path and
output-persistence fixes. No sample, weight, temperature, calibration, metric,
or threshold changed. This procedural retry is recorded; the cohort is now
observed development data and must not be treated as blind evidence.

Integrity checks on the saved run report:

- 1,000 examples; 562 speaker groups; exactly 100 examples in every class.
- Zero overlap with prior train/validation speakers, source-path identities,
  source rows, and exact WAV SHA-256 values.
- Zero duplicate normalized PCM payloads within the new cohort.
- Zero source validation/test shards or sealed audit records accessed.
- Historical raw WAVs are absent, so header-insensitive PCM overlap against
  the previous corpus is `UNKNOWN`; historical exact WAV hashes were checked.
- Thirty Parquet rowgroups contained 3,000 rows. About 86.3 MB compressed
  audio-column data was requested; 2,000 collateral rows were not persisted or
  scored. Only the 1,000 selected WAVs (30,982,680 bytes) were retained.

## Frozen candidate and measurement

- Whisper Tiny encoder:
  `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, weights SHA
  `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Existing text-free readout SHA
  `171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3` and
  pre-frozen temperature `0.42702477413437295`.
- 8,208,384 encoder + 3,850 readout = 8,212,234 resident parameters.
- The readout was trained on earlier speakers and the temperature was fitted
  on a separate earlier training-speaker calibration subset. No training,
  fitting, checkpoint selection, or temperature selection used this cohort.
- Ten supplied keyword options. Primary metrics: accuracy, NLL, multiclass
  Brier, ECE-15 and mean confidence. ECE uses 15 equal-width confidence bins;
  Brier is the per-example sum-of-squares averaged over examples.
- Baselines: uniform-10 and a one-hot training-majority class (`off`, selected
  only from earlier fit counts). Uncertainty uses 10,000 paired speaker-group
  bootstrap replicates, seed 17, two-sided percentile 95% intervals.
- Frozen config:
  `configs/pretrained_reuse/speech_commands_fresh_speakers_eval_v2.json`,
  SHA-256 `12d1e61985ff4f362fbf4ac43286865685a534941adfa7e7eeac495ba6e09b99`.

## Results

| Measure | Candidate | 95% speaker-group interval | Uniform-10 | Train-majority one-hot |
|---|---:|---:|---:|---:|
| Accuracy | 0.9380 | [0.9176, 0.9562] | 0.1000 | 0.1000 |
| NLL | 0.2290 | [0.1669, 0.3001] | 2.3026 | 24.8679 |
| Multiclass Brier | 0.0866 | [0.0637, 0.1130] | 0.9000 | 1.8000 |
| ECE-15 | 0.0237 | [0.0181, 0.0381] | 0.0000 | 0.9000 |
| Mean confidence | 0.9454 | — | 0.1000 | 1.0000 |

Per-class accuracy (100 examples each):

| Keyword | Accuracy |
|---|---:|
| yes | 0.98 |
| no | 0.91 |
| up | 0.98 |
| down | 0.92 |
| left | 0.94 |
| right | 0.93 |
| on | 0.92 |
| off | 0.92 |
| stop | 0.94 |
| go | 0.94 |

Candidate-minus-baseline paired bootstrap intervals were below zero for both
NLL and Brier against both controls:

| Baseline | NLL difference interval | Brier difference interval |
|---|---:|---:|
| Uniform-10 | [-2.1357, -2.0025] | [-0.8363, -0.7870] |
| Training-majority one-hot | [-25.1418, -24.0934] | [-1.7574, -1.6667] |

The candidate clears the sample-count, independent-group, ECE, NLL and Brier
conditions. It does **not** clear the accuracy condition: the point estimate is
93.8%, below 95%, and the lower confidence endpoint is 91.8%, below 95%. The
interval overlaps 95%, so this cohort does not establish that the underlying
speaker-population accuracy is below 95%; the provisional gate still fails.

The previous already-observed 640-example validation for this checkpoint had
96.56% accuracy, NLL 0.1435, Brier 0.0559 and ECE-15 0.0245. The fresh-speaker
transfer estimate is 2.76 percentage points lower, with higher NLL and Brier.
This is a descriptive comparison between separate cohorts, not a paired
significance test. The earlier validation is not reclassified as blind.

## Runtime, hashes and artifacts

- Local RTX 3080 Laptop, CUDA 12.4, Python 3.11.9, PyTorch 2.6.0+cu124,
  Transformers 5.6.2, safetensors 0.8.0.
- Model load: 0.248 s; 1,000 inferences: 12.008 s (83.27 examples/s); peak
  allocated CUDA memory: 60,880,896 bytes. No model training, cloud compute,
  or paid compute.
- Evaluation source commit: `14a9809e0e8a06c32708def0cd342df241c078c0`.
- Evaluator SHA-256:
  `64885d0dd7d448913b8dcc02d221cce666b33df736427b996feef22a7061c4f5`.
- Materializer SHA-256:
  `a12c251436c74ff53eb77eb77108e40699ab243a8cec4c4057334b218409bbd7`.
- Result directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-02-results`.
- Report SHA-256:
  `187937ea10611f5a64b00844480010a0c7f8eb4264681b5a8425b98bdf33d1e9`.
- Metrics SHA-256:
  `7ed4931f9dc5bbf49d14df8f568a5b6ce81db49bb86fbb627da12a7638af386d`.
- Predictions SHA-256:
  `06a0eea3386875717453ecc6535fc546d31026e1ef08c28482651d17c223f559`.
- Report time: 2026-11-10 01:18:28 JST. Training and bootstrap wall time were
  not separately instrumented; the measured inference interval above is exact.

## Decision and remaining work

Do not retune or retrain against this now-observed cohort. Keep this fixed
audio branch as narrow transfer evidence: its proper scores and calibration
are useful, but its accuracy falls short of the provisional 95% gate and the
previous validation point. This neither proves an audio capability ceiling
nor qualifies the model for the world-representation objective.

The broader objective remains open: text, image, video and genuine joint-input
Decision quality, complete product task coverage, and deployment/package gates
still need evidence. Continue with a separate frozen task/cohort and follow
the repository R0–R8 gates; do not pool this audio result with other
modalities or use it to claim 95% multimodal accuracy.
