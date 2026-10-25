# iNaturalist bounded metadata preflight (2026-10-26)

## Result

**Partial pass for schema and rights-field coverage; observer-disjoint corpus
feasibility is not established.** This was a bounded, metadata-only check of
the live iNaturalist API. No image bytes or example-level records were saved,
no personal identifiers were emitted to the report, and no model was trained.

The query was Aves (`taxon_id=3`), Research Grade, photos present, observation
license `cc0,cc-by`, and photo license `cc0,cc-by`, ordered by newest
observation creation time, with 100 observations requested. The live API
reported 4,284,912 matching observations for this request. The response had
100 unique observations and 36 unique observer IDs. Those counts are a
convenience sample, not an independent split and not a frozen dataset.

| Check | Observed |
|---|---:|
| Observation license field present | 100/100 (17 CC0, 83 CC BY) |
| Photos returned | 219 |
| Photo license field present | 219/219 (56 CC0, 163 CC BY) |
| Photo attribution field present | 219/219 |
| Unique observers in 100 observations | 36 |
| Media downloaded / retained | 0 |

The first five rows were also checked transiently to confirm the response
exposes observation `license_code`, photo `license_code` and `attribution`, and
an observer ID. Only field-presence summaries were retained. Observation
description/species-guess text was not inspected or stored.

## Interpretation and limits

The live filters and individual response fields appear sufficient to enforce
the separate observation/photo license gate and retain photo attribution for
this small sample. In a later admitted corpus, select at most one image per
observation and preserve its exact photo-level attribution/license receipt.
Use observer IDs only transiently to construct a deterministic grouped split;
do not expose login/name/location metadata to the model or public Git.

The 36 observers among the newest 100 records do not show that a sufficiently
large, balanced, observer-disjoint train/validation/sealed-audit corpus exists.
The convenience sample is time ordered and may be clustered. The live total
is dynamic (the earlier aggregate screen recorded 4,284,909), so neither count
is a reproducible corpus size. No species list, data export snapshot, split,
image hash, or normalized-content overlap test has been frozen. Do not download
media or fit a classifier based on this preflight alone.

Next gate: identify an authoritative, dated export with per-observation and
per-photo license/attribution evidence; run a metadata-only inventory for a
predeclared species shortlist; then measure observer-group and image/content
split feasibility before acquiring a bounded media sample. If the export does
not expose enough rights/observer fields, keep the source unadmitted.

## Method

The official API was queried three times with bounded responses (5, 1 and 100
observations); no request fetched media. The 100-row aggregate query did not
print or persist observation IDs, observer IDs, logins, coordinates, or raw
response rows. Rate guidance and bulk-export restrictions remain as recorded
in the [source screen](R2_INATURALIST_BIRD_IMAGE_SOURCE_SCREEN_20261026.md).
The API is live and no response hash is claimed.
