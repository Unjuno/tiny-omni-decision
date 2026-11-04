# iNaturalist CC0 pages 6–10 ascending coverage gate — 2026-11-04

## Result

The metadata-only expansion fails the previously frozen eligibility gate.
After excluding every observer, observation, and photo identity in the earlier
iNaturalist cohorts, including ascending pages 1–5, only **10 train and 6
validation** records remain. They cover Mallard (9/6) and Canada Goose (1/0);
the other six taxa contribute none. The fixed threshold remains 700 train / 200
validation overall, at least 25 train / 10 validation per taxon, and all eight
taxa represented. No media was downloaded, no features or metrics were
produced, and no checkpoint was trained or selected.

## Query and provenance

- Source: live iNaturalist API `/v1/observations`; retrieved
  `2026-11-04T12:58:36Z`.
- Query: eight previously used bird taxa; Research Grade; photos required;
  observation and photo licenses both explicitly CC0; `date_added` ascending;
  pages 6–10, 200 records per page; seed 17, observer-group split 70/30.
- 40 API pages returned exactly 8,000 observations. One cross-taxon photo
  identity was excluded, five observations had no eligible photo, and six photo
  objects lacked explicit CC0. The source freeze selected 1,930 observer/class
  rows before historical-exposure filtering (1,328 train / 602 validation).
- The existing expansion filter was run without modifications from source
  commit `e3909424a583c7fc5a216e7f277624b6403569d3`. It checked the original
  2026-10-26 cohort, all previously recorded chronological windows and the
  ascending pages 1–5 cohort. Historical observer overlap excluded 1,914 rows;
  zero train/validation observer, observation or photo overlap remains.
- The filter reports 2,730 previous observer hashes, `all_classes_represented`
  false, and `eligible_for_training` false. Do not use the 10/6 remainder for
  fitting or lower the threshold.

## Hashes and local artifacts

All source and filtered files remain outside Git under:

- Source: `C:\CodexArtifacts\inat-birds-cc0-ascending-pages-6-10-20261104-attempt-01\source\`
- Filtered result: `C:\CodexArtifacts\inat-birds-cc0-ascending-attempt-02-pages-6-10-20261104\`

| File | SHA-256 |
|---|---|
| Source `metadata.json` | `3e04464f3789ed37a039b0ae09f42855548a35c2d97dc76f08a763044364654e` |
| Source `train.jsonl` | `7d3efe534d26dfb62344067ed0dc6ce9ac61ebb1dae4f8196e80033f1202baaf` |
| Source `validation.jsonl` | `d4a6f7b85cc0f3304832c1abccf6a92609678b5bc94dd4f8def846208634501b` |
| Filtered `metadata.json` | `7fda0ba44231f8db663d400af042fe47b7a851f1bb19d83730aeabfa2c258b04` |
| Filtered `train.jsonl` | `5a6f26c51424ce3b1c5c5cc0a284cc38bc27a4dc7b0066d45fcacc87606ee4ee` |
| Filtered `validation.jsonl` | `e5e32703bf6705ccb29d1cc306caf00af11e45065ebbaf2e15be55ed3ddf5636` |

The filtered manifest contains hashes of every previous cohort and the new
source files. It retains pseudonymous observer-group hashes only; raw API rows,
observer names, coordinates and descriptions are not retained. The sealed audit
was neither materialized nor accessed.

## Decision

The ascending pages 1–10 plus the previous descending windows do not yield an
eligible fresh cohort for these same eight species. Do not continue deeper
pages for this cohort, select the 16 remaining records, or reuse previously
exposed validation to tune. A next image task needs a distinct predeclared
source/task with rights-clear media, sufficient independent groups and an
appropriate transfer split. This is a source-coverage failure, not evidence of
a VideoPrism representation ceiling.
