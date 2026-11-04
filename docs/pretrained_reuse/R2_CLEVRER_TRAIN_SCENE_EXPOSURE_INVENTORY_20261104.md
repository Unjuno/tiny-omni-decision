# CLEVRER TRAIN scene exposure inventory — 2026-11-04

## Decision

This read-only metadata audit does **not** establish a fresh CLEVRER evaluation pool suitable for another fit. Do not start a new CLEVRER experiment from this inventory. The `2,837` scenes absent from the scanned corpus manifests are only an upper bound, not verified unused scenes.

## Scope and method

- Source revision: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`; official TRAIN reference count: 10,000 scene IDs.
- Scanned 118 previously preserved corpus/readout JSONL manifests under `C:\CodexArtifacts\pretrained-reuse-r2\clevrer*`, totaling 98,004 rows. The identity parser recognized 59,438 rows and 7,163 unique official TRAIN scene IDs (IDs 1–9,999).
- `test`, `sealed`, `audit`, and `holdout` path components were excluded. No media was opened. The one-time CLEVRER Fresh Reasoning holdout data was not read; its existing status report says it contains 200 scenes and has already been evaluated.
- Per-file byte counts and SHA-256 hashes are in the external receipt `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-train-exposure-inventory-20261104\summary.json` (SHA-256 `13434a06917f52a3d118e8d84c1cc7d627d51ded6362fcbc0d4a35382d35d575`). The receipt's manifest-inventory hash is `2728f013a855b76dac757554f97fd5e5daacc5d83db12ba30267c18c1d1d4947`.

## Reconciliation findings

- The historical tracked exclusion manifest contains 307 TRAIN and 212 validation IDs, with zero cross-split overlap. All 307 historical TRAIN IDs occur in the V2 TRAIN exclusion set, and all 212 historical validation IDs occur in its validation exclusion set; no cross-split migration was found.
- The V2 coverage experiment consists of 256 baseline TRAIN scenes plus 512 additional TRAIN scenes (768 total) and 256 validation scenes. The additional TRAIN and validation sets have zero scene overlap; both are disjoint from their corresponding corrected exclusion sets. Hashes match the V2 result report.
- Across the selected corpus manifests, 7,163 of 10,000 official TRAIN IDs are represented. `10,000 − 7,163 = 2,837` is **not** a count of unused data: the scan intentionally omitted holdout/test/audit paths and does not cover every historical receipt or every metadata-only source inspection.
- The existing synthetic AV source report says 405 scenes from its previously unseen candidate window were fetched and integrity-checked; 401 had eligible rows and were used in its 322-scene train / 79-scene development-validation split. The separate Fresh Reasoning holdout was also evaluated once. These previously used groups cannot be treated as a new blind validation set.

## Operational result

No training, inference, data acquisition, model load, validation re-evaluation, or sealed/test access occurred. No existing dataset, run, checkpoint, or worktree was changed. The audit scripts and receipt are small and retained as reproducibility artifacts; legacy model/data artifacts remain untouched. A new CLEVRER experiment needs a separately proven, sufficiently large, independent evaluation pool before fitting.

## Reproduction

The external directory contains `scan.py`, `reconcile.py`, and `summary.json`. `scan.py` reads JSONL scene identities and hashes only, while excluding test/sealed/audit/holdout paths. `reconcile.py` compares the scan with the historical and V2 split manifests. The SHA-256 of each executable script is recorded beside the receipt in the inventory directory.

Primary tracked references: `manifests/pretrained_reuse/clevrer-heldout-video-excluded-scenes-20261011.json`, `docs/pretrained_reuse/CLEVRER_UNIQUE_COVERAGE_V2_PROTOCOL.md`, `docs/pretrained_reuse/CLEVRER_UNIQUE_COVERAGE_V2_RESULTS.md`, `docs/pretrained_reuse/R2_CLEVRER_SYNTHETIC_AV_RESULTS_20261020.md`, and `C:\CodexWorktrees\r2-clevrer-fresh-reasoning-20261021\docs\pretrained_reuse\CLEVRER_FRESH_REASONING_V1_STATUS_20261021.md`.
