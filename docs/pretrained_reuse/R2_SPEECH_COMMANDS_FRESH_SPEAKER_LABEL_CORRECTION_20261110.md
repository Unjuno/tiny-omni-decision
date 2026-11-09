# Speech Commands label correction (2026-11-10)

## Finding

The first metadata-only cohort selector incorrectly treated numeric label IDs
as alphabetical class order. For example, selected Parquet row
`train-00005-of-00006.parquet:7703` has numeric label `0` and source file
`yes/d8c314c0_nohash_0.wav`, while the first manifest had target `down`.
The pinned Google Speech Commands v0.02 dataset card defines numeric labels in
the order `yes, no, up, down, left, right, on, off, stop, go`; the original
Speech Commands corpus manifest likewise records label `0` as `yes`.

The first evaluation report and predictions remain unchanged at
`C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-01-results`.
Its supplemental `evaluation-validity-audit.json` marks the result invalid for
quality claims because targets were permuted. No checkpoint, temperature,
sample IDs, audio bytes, or model settings were changed in response to those
metrics.

## Corrected frozen cohort

The correction re-read only `label`, `speaker_id`, and `file` columns from the
same pinned TRAIN rowgroups, then verified every numeric ID against the source
path class and prior selected-row identities. It did not request the Parquet
audio column. It retained exactly the same 1,000 sample IDs in the same order,
the same 562 speakers, and the same 1,000 previously verified WAV byte hashes.
Every one of the ten classes remains at 100 examples.

- Prior manifest hash: `96c6fe86298ef234e3b49546df8e07858049283a1e1334e4982a5bdbad5daaa4`.
- Corrected manifest hash:
  `06baea7a101ce60849b5a6dc81d4413c531c0871091fa914d169a53e5f30f23e`.
- Corrected inventory hash:
  `83dd38c4247666fb4b3d3a18e22da45cd9e8ab3c3343296a53ae3160be10c816`.
- Correction artifact:
  `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-02-corrected`.

The corrected manifest, class map, checkpoint hash, calibration temperature,
metric definitions, bootstrap, controls, and acceptance thresholds are frozen
in `configs/pretrained_reuse/speech_commands_fresh_speakers_eval_v2.json`.
Attempt 02 is the only quality evaluation eligible for interpretation. The
earlier invalid run is never used for candidate selection or tuning.

## Remaining data limitation

No raw WAV files from the old 1,920-example corpus are retained in its
historical directory. Exact historical WAV hashes, source path hashes, source
row IDs, and speaker-group hashes were compared and have zero overlap. A
header-insensitive normalized-PCM comparison against the prior files is
therefore unavailable; within the new cohort the evaluator hashes normalized
PCM and reports duplicates.

## Sources

- [Pinned Speech Commands dataset card](https://huggingface.co/datasets/google/speech_commands/blob/a751309c0fd613e8a5d30d77900f30e8b42bc2da/README.md)
- [Speech Commands paper](https://arxiv.org/abs/1804.03209)
