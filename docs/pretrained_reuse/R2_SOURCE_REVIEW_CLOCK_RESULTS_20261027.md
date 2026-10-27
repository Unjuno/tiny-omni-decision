# Frozen source-review sample: container clock preflight

The [fixed protocol](R2_SOURCE_REVIEW_CLOCK_PROTOCOL_20261027.md) checked all
16 previously selected train-source videos against the saved acquisition SHA.
All 16 annotated event intervals were covered by contiguous decoded audio
sample blocks using the repository's exact half-open interval function; all
16 also had at least one decoded video frame with a PTS inside the same event
interval. The audio sample rates were 44,100 or 48,000 Hz. First decoded audio
PTS was zero in each file, in the respective sample-rate time base. The receipt
retains every file hash, requested interval, actual video PTS, audio sample
bounds and PyAV version (18.1.0).

This establishes only container timestamp and decoded-sample coverage for this
small, deliberately stratified sample. It does **not** demonstrate physical
synchronization, confirm that the annotated sound is audible, identify the
cause of a sound, review the complete video, or validate the annotation. The
sample is not a prevalence estimate, and the earlier source-label semantic gate
remains open. A reviewer must still examine the synchronized audio/video event
and context. No model, additional data acquisition, validation split or sealed
audit was used.

Result:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-review-clock-20261027-attempt-01/result.json`,
SHA-256 `2297e678ca9bbc39a2c16d0ef3176ccfcd7d43f440fcd42c82db6f00c31e892c`.
Reproduce from the worktree with `PYTHONPATH=src;.` and
`python -m scripts.audit_source_review_clocks --sample <frozen-sample.json> --output <fresh-result.json>`.
