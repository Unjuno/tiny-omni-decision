# CLEVR-4 spatial-readout reporting-set exposure audit — 2026-11-13

## Disposition

**Do not score the current 4×4 spatial-readout candidate on its designated
reporting partition as fresh or blind evidence.** An earlier fixed-readout run
already generated predictions for exactly the same 5,296 image IDs. The
current spatial run correctly records `reporting_partition_loaded=false`;
keep it that way. The existing one-time reporting set is no longer an
untouched comparison set for a new candidate.

## ID-only reconciliation

The ID join used only the `asset_id` field from the current report manifest and
the `image_id` field from the prior predictions. A schema check emitted the
first prior prediction row to the console, including its target and probability
list; no aggregate metric was recomputed and no output was used for fitting or
selection. No media, feature arrays, checkpoints, or sealed-audit data were
opened.

- Current candidate: `path-a-image-videomae-clevr4-spatial-grid-4x4-seed17`.
- Current run metadata says `reporting_partition_loaded=false`.
- Current report manifest:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03\report-media-manifest.jsonl`
  SHA-256 `0dd87a1303cb25bbc87ed3d69a2acde5ecef02f27bec16985297e9d685b09525`.
- Prior one-time predictions:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-decision-fit-20261110-attempt-02\reporting-one-time-predictions.jsonl`
  SHA-256 `4c39f57964c4440c7f2e71a5c35da9652459d14d99b8fc2634e603bda084461e`.
- Prior selected checkpoint SHA-256:
  `437e920e7cda02ff2e28f19c96433af04aab3d70494030b6e3278809aa1cc39e`.
- Prior reporting was completed at `2026-11-09T23:19:03.254644+00:00`.
- Current manifest: 5,296 distinct image IDs. Prior predictions: 21,184 rows
  over 5,296 distinct image IDs. Intersection: 5,296; current-only: 0;
  prior-only: 0.
- SHA-256 of the sorted unique image-ID set:
  `089fa00e1704f1150f87a219ba59bafbe79d1d067ccc255dd537fd1677622cb0`.
- Machine-readable receipt outside Git:
  `D:\CodexArtifacts\pretrained-reuse-r2\clevr4-report-exposure-audit-20261113-attempt-01\id-overlap-receipt.json`
  SHA-256 `9b96ad6cfcc4ddfff94462bfa3bfcb285a0ff44db351f11ecb8dd4ead6bdfcb0`.

## Consequence

The current candidate has only its already-observed 1,000-image selector
result. Do not use the shared 5,296-image reporting partition to choose,
compare, or promote this or another readout. A new image-quality evaluation
needs a separately sourced, source-disjoint cohort or must remain an
explicitly development-only comparison. Preserve both existing checkpoints
and all predictions unchanged.
