# Fixed visual-LSTM frame-order intervention

The [fixed protocol](R2_SOURCE_VIDEO_LSTM_ORDER_PROTOCOL_20261027.md) reloaded
the three saved train-only OOF visual-LSTM checkpoints. Original predictions
were re-computed and matched their saved probabilities, labels, decisions, NLL
and Brier before a frame-order intervention was accepted. No new training,
old development/official validation, test, or sealed audit was used.

Only the real frame sequence supplied to the LSTM residual was reversed.
The original mean/last-minus-first visual path and audio input stayed unchanged;
padding was not reversed and no new frames were created. This is not a full
time reversal of the input and cannot establish temporal understanding.

| Subset | Records | Original accuracy | LSTM sequence reversed | Changed choices |
|---|---:|---:|---:|---:|
| All | 2,593 | 0.5534 | 0.5476 | 157 |
| At least one multi-frame candidate | 1,179 | 0.5768 | 0.5640 | 157 |
| Visible source with a multi-frame correct candidate | 650 | 0.5015 | 0.5123 | 105 |

Across all records, original minus reversed Accuracy was +0.58 percentage
points, with a paired video-group bootstrap interval −0.14 to +1.34 points;
the set-NLL difference was −0.0132 (interval −0.0318 to +0.0056). Bootstrap
used 10,000 draws, seed 17, over 320 video groups. This observed order
sensitivity changes some selections but does not establish a consistent
aggregate improvement. The positive-multiframe subset is a descriptive
condition, not an independently selected benchmark. The data and models are
nonblind, folds share training data, only one seed was used, and source labels,
physical timing and participant independence remain unresolved.

The CPU-only receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-video-lstm-order-20261027-attempt-01/result.json`,
SHA-256 `306720384e985d24c918df5b264dcda09249bdba54e796745533b844cc8cedcf`.
It records protocol, script, source-run and per-record output hashes. Four
targeted tests and Ruff passed; the preceding temporal training commit's CI
passed. CUDA was not initialized. These results neither promote a model nor
change the four-modality quality objective.
