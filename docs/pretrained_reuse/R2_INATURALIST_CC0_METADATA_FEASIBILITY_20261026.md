# iNaturalist strict-CC0 bird metadata feasibility (2026-10-26)

## Disposition

**Continue as a candidate; do not admit media or claim a quality result yet.** This was a live-API metadata-only screen. No image bytes, raw observation rows, photo IDs, observer IDs/names, coordinates, or labels for individual examples were retained. No model was trained or evaluated, and no sealed set was materialized or opened.

Strict CC0-only results show enough current candidate volume for a bounded eight-class image-only Decision probe. The main risk is observer clustering: record counts substantially overstate independent examples for some taxa. A later fixed cohort must be grouped globally by observer and media identity, with a one-observation-per-observer-per-class cap for evaluation or a clearly defined group-weighted analysis.

## Query and source constraints

On 2026-10-26, eight predeclared bird taxa were queried from the public iNaturalist API with `quality_grade=research`, `photos=true`, `license=CC0`, `photo_license=CC0`, `per_page=200`, `order_by=date_added`, and `order=desc`. Five pages (1,000 observation rows) were aggregated per taxon. The query was run at the API's documented approximate one-request-per-second pace. Only counts were emitted.

The first 1,000 rows per class are a time-ordered convenience sample, not a frozen corpus. `total_results` is a live count. The API docs recommend the supported `date_added` sort; the earlier `order_by=id` preflight is not used as an ordering guarantee. The official API recommends small/medium retrieval and an approximate one-request-per-second rate, not bulk crawling. This screen does not use the weekly GBIF export. A prior review recorded a dataset-level CC BY-NC declaration for that separate export; this screen did not recheck it.

The API response has separate observation and photo license fields. The API's strict CC0 filters did not remove every photo object lacking an explicit `cc0` photo field: one photo object in the 1,000-row sample was non-CC0 or unlabeled. A future selector must verify every chosen observation's `license_code` and every chosen `photos[].license_code` directly, and exclude any non-CC0, missing, conflicting, or revoked item. Only images from CC0 photos paired with CC0 observations are in scope. No record is admitted by this aggregate screen alone.

## Aggregate results

The provisional observer split hashes the observer identifier in memory with SHA-256 over `seed17|observer_id`, interprets the first four digest bytes as a little-endian unsigned integer, takes modulo 100, and assigns values `<70` to train, `<85` to validation, and the rest to an audit role. No identifiers or assignments were saved. Counts below are feasibility diagnostics only; this is not a frozen split. `one-per-class-group` counts are the unique observers in each split and are the maximum independent examples if later capped to one observation per observer per taxon.

| Taxon | ID | Live CC0-filtered total | Sampled rows | Unique observers | Train groups | Validation groups | Audit-role groups | Raw rows train/val/audit | Photo objects not explicitly CC0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mallard | 6930 | 15,046 | 1,000 | 287 | 202 | 44 | 41 | 763 / 144 / 93 | 0 |
| House Sparrow | 13858 | 7,310 | 1,000 | 373 | 257 | 54 | 62 | 709 / 143 / 148 | 0 |
| Canada Goose | 7089 | 8,644 | 1,000 | 332 | 232 | 51 | 49 | 696 / 195 / 109 | 0 |
| Rock Pigeon | 3017 | 7,486 | 1,000 | 308 | 222 | 38 | 48 | 728 / 165 / 107 | 0 |
| American Robin | 12727 | 7,333 | 1,000 | 278 | 204 | 36 | 38 | 705 / 176 / 119 | 0 |
| Great Egret | 144455 | 5,624 | 1,000 | 337 | 233 | 54 | 50 | 701 / 146 / 153 | 1 |
| Great Blue Heron | 4956 | 7,657 | 1,000 | 254 | 179 | 36 | 39 | 613 / 284 / 103 | 0 |
| Northern Cardinal | 9083 | 6,977 | 1,000 | 156 | 113 | 20 | 23 | 221 / 737 / 42 | 0 |
| **Total / unique** | — | — | **8,000** | **1,033** | **729** | **146** | **158** | **5,136 / 1,990 / 874** | **1** |

The raw row split is highly imbalanced for Northern Cardinal because a small number of observers contribute many records. The unique-observer counts are the appropriate independent units; even the smallest audit-role class has only 23 unique observers in this convenience sample. This is too small to support a precise 95% accuracy claim. The one photo ID repeated across two taxa is a cross-class asset collision and all candidate rows sharing it must be quarantined until resolved. Observation IDs were unique across taxa in the sample.

The sample contained 12,597 CC0 photo objects before global deduplication, 12,596 unique photo IDs, and one cross-taxon repeated photo-ID group. These are metadata counts; no media existence check, media SHA-256, perceptual-hash check, or normalized-content overlap test has run.

## Interpretation and next gate

The strict-CC0 candidate is more suitable than widening to CC-BY for this first probe, but the current result does not establish legal/runtime admission or natural-image grounding. The official iNaturalist Open Dataset README says observation and photo records are distinct, metadata snapshots are generated monthly, and each photo's own license governs reuse. The earlier screen's GBIF weekly-export license concern remains relevant only if that export is used; this experiment would query the live iNaturalist API directly and must preserve the exact returned per-record rights evidence.

Before acquiring media:

1. Freeze a fresh exact metadata receipt for a deterministic, balanced shortlist and verify all observation/photo licenses individually.
2. Select at most one image per observation; quarantine the repeated photo ID and any missing/non-CC0 license.
3. Group globally by observer, keep source/asset/media identities disjoint, and cap or report repeated-observer influence. Increase the cohort or revise the class shortlist based only on predeclared availability/group-count rules if the independent validation/audit counts remain too small.
4. Save the selected IDs, retrieval time, raw-metadata hash, split hash, and later media hashes outside public Git; retain no coordinates or observer display names in the model inputs. No media was downloaded during this screen; acquisition remains deferred until the individual-record and split gates pass.
5. Only after the frozen data gate passes, run the existing prequantized VideoPrism-B image path with a small typed classification head, using training/validation only for selection. Keep a new sealed group-disjoint audit closed until selection is frozen. Use the HotUGPU guard before any GPU work.

The task, if admitted, is an eight-option bird-species image classification probe. Its score cannot be generalized to arbitrary image decisions or the four-modality product target. The 95–98% aspiration remains unverified.

## Official references

- [iNaturalist API](https://www.inaturalist.org/api) — separate observation and photo license filters; supported sort fields.
- [API recommended practices](https://www.inaturalist.org/pages/api%2Brecommended%2Bpractices) — small/medium request scope, pagination and rate guidance.
- [iNaturalist licenses for observations, photos, and sounds](https://help.inaturalist.org/en/support/solutions/articles/151000175695).
- [iNaturalist Open Dataset README](https://github.com/inaturalist/inaturalist-open-data/blob/main/README.md?plain=1) — monthly metadata snapshots, distinct photo licensing, and attribution fields.
- Repository [dataset source and license policy](../../DATASETS.md#license-policy).
