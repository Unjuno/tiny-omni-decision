# Speech Commands fresh-cohort relationship audit — 2026-11-10

## Scope

This is a read-only comparison of the metadata for two already completed,
already observed development evaluations. It did not read WAV payloads, load a
model, rerun inference, fit parameters, or access validation/test shards or a
sealed audit. It answers only whether the two reported cohorts share source
examples or speaker/media identities; it is not a new quality result.

Both reports use `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`
TRAIN data and the same ten fixed English keywords, but they evaluate different
readouts and different cohorts:

- M1 fresh-speaker readout: 1,000 validation examples / 200 speakers; 95.1%
  accuracy, NLL 0.1655, Brier 0.0654, ECE-15 0.0225. See
  [`R2_SPEECH_COMMANDS_FRESH_M1_RESULTS_20261110.md`](R2_SPEECH_COMMANDS_FRESH_M1_RESULTS_20261110.md).
- Frozen-readout fresh-speaker transfer: 1,000 examples / 562 speakers; 93.8%
  accuracy (speaker-group bootstrap 95% interval 91.8–95.6%), NLL 0.2290,
  Brier 0.0866, ECE-15 0.0237. See
  [`R2_SPEECH_COMMANDS_FRESH_SPEAKER_RESULTS_20261110.md`](R2_SPEECH_COMMANDS_FRESH_SPEAKER_RESULTS_20261110.md).

These are not a paired model comparison: the readout weights differ, and the
cohorts differ. They are also not independent seed replications. They are
disjoint speaker/example cohorts for the same narrow keyword task and source
revision.

## Metadata inputs and integrity

Only these JSONL files were parsed:

| Role | Path | Rows | SHA-256 |
|---|---|---:|---|
| M1 validation manifest | `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-media-20261110-attempt-04\validation.jsonl` | 1,000 | `e19e5c44b6c3fc01d247bcd163dfbe1f46b6e95bc59ccbb6d800f90de3ea3ec9` |
| M1 full train/validation media inventory | `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-media-20261110-attempt-04\media-inventory.jsonl` | 2,500 | `206b626ae87cbbac9f2d1b0b9d5568ff4f21ae9186aa87b2f1c530da71134b49` |
| Corrected transfer cohort manifest | `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-02-corrected\cohort-manifest.jsonl` | 1,000 | `06baea7a101ce60849b5a6dc81d4413c531c0871091fa914d169a53e5f30f23e` |
| Corrected transfer media inventory | `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-02-corrected\media-inventory.jsonl` | 1,000 | `83dd38c4247666fb4b3d3a18e22da45cd9e8ab3c3343296a53ae3160be10c816` |

Shard-local row numbers were compared as `(source_file, source_row)` pairs,
not as bare row numbers. Equality sets were then compared for source-path
SHA-256, speaker-group SHA-256, and exact audio SHA-256 (`media_sha256` in the
M1 validation manifest and `audio_sha256` in the media inventories).

| Comparison | Source-file/row | Source-path hash | Speaker hash | Exact audio hash |
|---|---:|---:|---:|---:|
| M1 validation vs corrected transfer | 0 | 0 | 0 | 0 |
| Entire M1 2,500-row inventory vs corrected transfer | 0 | 0 | 0 | 0 |

The two example cohorts are therefore disjoint on the available source-row,
speaker, path, and exact-byte audio identities. This check does not prove
header-insensitive PCM equivalence beyond the exact audio hashes represented in
the inventories; the transfer report separately records historical PCM
overlap as unknown where historical raw WAVs are absent.

## Interpretation

Keep both measurements in the record without selecting the higher point
estimate as the result. The 95.1% M1 point estimate is on its own 200-speaker
cohort; the separate fixed-readout transfer cohort is 93.8% and does not pass
the provisional 95% accuracy gate. The observed difference is descriptive only
because the model weights differ and the cohorts are unpaired. Neither result
establishes broad speech quality, the other required modalities, joint-input
quality, or product readiness.
