# Speech Commands official validation remainder screen — 2026-11-11

## Decision

Do not materialize audio or evaluate the remaining official-validation rows.
After excluding every speaker group in the prior Path A validation sample, the
remaining pinned validation shard has only 222 keyword rows across 39 speaker
groups. This is below the frozen minimum of 1,000 rows and 200 independent
groups for a fresh transfer readout.

## Metadata-only method

- Source: `google/speech_commands`
- Revision: `a751309c0fd613e8a5d30d77900f30e8b42bc2da`
- File: `validation-00000-of-00001.parquet`
- Expected full-file SHA-256: `06683ef0761319a415ee516939c64b9b7a20be0b6bb17ce8d2b6e36abe373a46`
- Full-file SHA-256 verified: **no**; only pinned Parquet metadata/ranges were
  read.
- Columns read: `label`, `speaker_id`, `file`; audio payload was not read.
- Shard metadata: 9,982 rows, 100 row groups; 91 range requests / 632,727
  transferred bytes.
- Prior Path A validation excluded: 640 rows / 206 speaker groups.
- Eligible remainder: 222 rows / 39 speaker groups. Per-keyword row counts are
  16–29 and unique-group counts are 12–20.

The predeclared sampling requirement is 100 examples per keyword, at most five
examples per speaker, and at least 200 independent speaker groups. The observed
remainder cannot meet it. No cohort was frozen, no audio was materialized, and
no inference or fitting was performed. The official test shard and sealed audit
were not accessed.

## Interpretation

This is a data-availability gate, not a model failure and not evidence that the
model cannot improve. The fresh-speaker Speech Commands result remains a narrow
closed-set keyword result; this remainder cannot provide a sufficiently powered
independent follow-up. Do not relax the group/sample thresholds or tune on the
already observed validation IDs to make another run possible.

Machine-readable details: `R2_SPEECH_COMMANDS_OFFICIAL_VALIDATION_REMAINDER_SCREEN_20261111.json`.
