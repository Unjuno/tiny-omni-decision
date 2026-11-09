# Fresh-speaker Speech Commands transfer protocol (2026-11-10)

Status: frozen before media acquisition and before inference on this cohort.
No selected-cohort audio has been downloaded or scored at the time this
protocol is committed.

## Question and scope

Does the existing frozen Path A text-free audio classifier transfer to
previously unused speakers from Speech Commands v0.02 official TRAIN? This is
one fixed-checkpoint, one-time development transfer evaluation for the narrow
10-way English keyword task. It is not an official test or sealed audit, does
not measure general speech understanding, and is not evidence of multimodal
quality.

## Pinned source and cohort

- Dataset: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`,
  config `v0.02`, official TRAIN Parquet shards only.
- The pinned dataset card identifies CC BY 4.0 and asks users not to attempt
  speaker identification. The original paper also identifies CC BY 4.0 and
  describes the dataset as a limited-vocabulary keyword spotting corpus.
- Attribution: Pete Warden, *Speech Commands: A Dataset for Limited-Vocabulary
  Speech Recognition* (2018), https://arxiv.org/abs/1804.03209; Google Speech
  Commands v0.02, CC BY 4.0.
- Frozen selected-row manifest:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-preflight-20261110-attempt-04-results\cohort-metadata.jsonl`,
  SHA-256 `96c6fe86298ef234e3b49546df8e07858049283a1e1334e4982a5bdbad5daaa4`.
- Metadata-only preflight: 1,000 examples, exactly 100 per class, 562 unique
  speaker groups, and maximum five examples per speaker. Selection seed 17.
- The prior Speech Commands fit and validation exposure (1,920 records,
  1,035 speakers) was excluded by source row, speaker identity, and asset path
  hash. Cohort identities and all ten labels were frozen before media retrieval.
- No official validation or test shard, prior official validation payload, or
  sealed audit data is part of this run.

The 30 required Parquet row groups contain 3,000 rows. Reading the Parquet
audio column therefore entails about 86,345,964 compressed bytes and 2,000
collateral rows. Only the 1,000 selected WAVs may be persisted or evaluated;
collateral row audio is neither written nor scored. The source-row, speaker,
path, label, split and WAV identities must be verified during materialization.

## Locked candidate

- Existing calibrated Path A readout only; no training, fitting, model choice,
  or temperature selection on the frozen cohort.
- Readout:
  `C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-calibration-03\last-readout.pt`,
  SHA-256 `171f6a9303b2fd9ad63efe2bee6af119549fb4f26ea663e9c1297f51a678e1c3`.
- Frozen temperature: `0.42702477413437295`, fitted on speaker-held-out
  partitions from previously used official TRAIN records, before this cohort
  was created.
- Encoder: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`,
  encoder weights SHA-256
  `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Exactly ten supplied class options in this order: yes, no, up, down, left,
  right, on, off, stop, go. The runtime returns a normalized probability JSON.

## Metrics and decision rule

Report accuracy, NLL, multiclass Brier (mean per-example sum of squared
probability errors), ECE-15, mean confidence, per-class recall, and the full
confusion matrix. Save every example's stable ID, hashed speaker group, target,
prediction, probability vector and media hash.

Compare against a uniform-10 distribution and a one-hot training-majority
baseline. The latter is chosen from the pre-existing fit partition's class
counts (ties broken by the frozen class order); it is not estimated from this
cohort. Use 10,000 paired speaker-group bootstrap replicates, seed 17, and
two-sided percentile 95% intervals. The candidate-minus-baseline intervals
for NLL and Brier are paired by resampling speaker groups, retaining all rows
from each sampled group.

Apply the provisional R0 gate in
`R0_QUANTITATIVE_ACCEPTANCE_POLICY_20261031.md`: at least 1,000 examples and
200 independent groups; accuracy point estimate and lower 95% group-bootstrap
endpoint at least 95%; ECE-15 point estimate and upper endpoint at most 0.05;
NLL and Brier each improve over both baselines with paired 95% upper endpoints
below zero. The policy is provisional and does not qualify a general product
release. No threshold, model, or cohort may be changed in response to the
result.

## Execution record

The one-time evaluator is
`scripts/evaluate_speech_commands_fresh_speakers.py`. The media materializer
is `scripts/fetch_speech_commands_fresh_speakers.mjs`; it requires the pinned
MIT-licensed `hyparquet` 1.31.3 parser and verifies the frozen cohort hash.
Evaluation output belongs under a new, empty directory outside Git. The
result report must record environment, runtime, hashes, resource use and the
provisional gate disposition. Preserve this protocol and the legacy audio
evaluation unchanged.

## Sources

- [Pinned Speech Commands dataset card](https://huggingface.co/datasets/google/speech_commands/blob/a751309c0fd613e8a5d30d77900f30e8b42bc2da/README.md)
- [Speech Commands paper](https://arxiv.org/abs/1804.03209)
- [R0 provisional quantitative acceptance policy](R0_QUANTITATIVE_ACCEPTANCE_POLICY_20261031.md)
- [Path A evaluation protocol](EVAL_PROTOCOL.md)
