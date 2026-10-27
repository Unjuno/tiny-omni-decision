# Fixed train-source annotation review sample

Purpose: examine whether the custom instruction and source annotations support
an AV task. Select four records each from `none_related_candidate`,
`none_other`, `visible_one` and `visible_multi` within the 2,593 retained
training-source records. Determine categories from the pinned source archive
and independently reconstructed candidate identities. Sort by
SHA-256(`source-label-review-v1:17:<record_id>`) then record ID, taking the
first records from distinct videos globally, processing categories in the
order listed above. No model outputs, scores, difficulty, or media contents
influence selection. Do not replace unavailable records; report failure.
This deliberately stratified 16-record convenience diagnostic is not a
statistically representative prevalence estimate or a blind product test.

Use existing official-train MP4 files and their existing acquisition manifest,
verify SHA-256 and timestamps, store review IDs, provenance, selected media
hashes and category counts. Do not copy or edit original media. For review,
inspect the full relevant event interval with audio and sufficient context;
still frames or annotations alone cannot establish a causal sound source.
Distinguish visible related object, visible contact point, unknown source,
multiple related objects and ambiguous audio. Do not relabel training data or
claim a corrected benchmark from a few examples. No official validation, sealed
audit, new acquisition, training or model selection.
