# Train-source annotation review: frozen sample and partial visual inspection

The [fixed sampling protocol](R2_SOURCE_SEMANTICS_REVIEW_PROTOCOL_20261027.md)
selected 16 train-source records, four each from the four annotation strata,
with distinct videos and no model-output-dependent selection. The 2,593-record
population comprises 30 none-visible records with a known related candidate,
662 other none-visible, 799 visible-one and 1,102 visible-multiple records.
All 16 original train MP4 files matched their existing acquisition manifest
SHA-256; no media was downloaded or copied. This is a stratified diagnostic,
not a representative prevalence estimate or independent validation set.

For the four `none_related_candidate` examples, three event-time video frames
per example were decoded and inspected. The frames show objects, hands or scene
content, but no claim about the audible event, causal source or visibility of
the sound-producing contact follows from these stills. The remaining 12 examples
and full video/audio sequences were not reviewed. In particular, this inspection
does not resolve the annotation-semantic gate, verify labels or justify changes
to the frozen corpus. A reviewer with synchronized audio and video must complete
the fixed protocol before task-specific training resumes.

Frozen sample:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-semantics-review-sample-20261027-attempt-01/result.json`,
SHA-256 `9d7df2419a4b3b63b23d1da94f571cf2ba29f3492b433caad0d530c756531126`.
The four-example still montage is
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-semantics-review-sample-20261027-attempt-01/none-related-frames-repro.png`,
SHA-256 `c860f17e13c27883050b0597a88976936b27c6e5b027222801b56bc473130d71`.
Reproduce with `python -m scripts.render_source_semantics_review --sample <frozen-result.json> --output <new-png>`.
No validation, sealed audit, model inference or training was performed.
