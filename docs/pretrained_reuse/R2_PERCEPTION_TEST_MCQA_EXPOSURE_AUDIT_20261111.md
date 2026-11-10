# Perception Test MCQA exposure audit — 2026-11-11

## Decision

This is a metadata-only exposure preflight, not a source admission and not a
quality result. It found a candidate video-grouped validation pool large enough
to justify a separate task-design review. It does **not** establish that the
candidate videos or questions are unseen, and it does not authorize fitting or
scoring on this branch. The official test split remains unopened.

## Scope and provenance

- Source: official Perception Test train and public-validation annotation
  archives, pinned in
  [`R2_PERCEPTION_TEST_SOURCE_SCREEN_20261111.md`](R2_PERCEPTION_TEST_SOURCE_SCREEN_20261111.md).
- Train archive SHA-256:
  `9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
- Validation archive SHA-256:
  `eeefeb43955b1850b5e521ad774fe66675baac0a2a06617bd83ffbe43aaf03d7`.
- Train has 2,184 video groups / 7,392 MCQA rows; validation has 5,900 groups /
  19,140 rows. The official train/validation video-ID overlap is zero.
- Artifact roots scanned: `C:\CodexArtifacts\pretrained-reuse-r2` and
  `D:\CodexArtifacts\pretrained-reuse-r2`. The inventory reads path names and
  small text/JSON metadata only; it does not open media, model weights, feature
  arrays, or the official test split. No inference, fitting, or metric
  evaluation was run.

## Corrected interpretation of the broad ID inventory

The first conservative scan found 8,085 video IDs in prior Perception-prefixed
artifacts. All 2,184 train and 5,900 validation IDs appeared somewhere, so a
literal “any mention means model exposure” rule would leave no candidates.
That rule over-counts exposure: the full 5,900-ID validation cut-frame mapping
is a source index, not evidence that every video was selected or scored.

The accompanying `perception-source-candidate-validation-transfer` corpus
manifest was read to resolve this point. It records 200 selected validation
videos, all held, with `model_loaded: false` and `metrics_computed: false`.
Thus neither the 5,900-ID source mapping nor this transfer artifact proves
validation model exposure for all rows. The manifest SHA-256 is
`4bbb357ac59f57543a4984eddc8d655bcf307cc0b705676e69d75b2d14150914`.

The scanner now keeps the conservative raw inventory and separately identifies
IDs with direct local paths for media, features, predictions, or named selected
cohorts. This direct-path inventory matched 1,939 train groups and 1,270
validation groups. The remaining 4,630 validation groups (13,994 MCQA rows)
are only **candidates without a direct artifact-path match**. Missing paths do
not prove that a model or person has never seen an item; incomplete inventories
and prior work outside the scanned roots remain possible.

The scanner excludes its own `perception-test-mcqa-exposure-audit-*` output
directories to prevent previous audit reports from becoming self-referential
exposure evidence. The final scan covered 248 candidate directories on C: and
2 on D:.

## Exact normalized-content overlap

To check repeated question/choice strings, the audit hashes the question and
three options after Unicode NFKC normalization, case folding, whitespace
collapse, and option-order normalization. The answer label is excluded. It
removes validation rows whose normalized content appears in any official train
question while keeping every video assigned to its original split.

Among the 4,630 path-screened candidate validation videos, this leaves 1,518
video groups with 2,886 questions. The remaining question counts by publisher
reasoning type are:

| Reasoning type | Questions | Video groups |
|---|---:|---:|
| Descriptive | 2,281 | 1,197 |
| Explanatory | 295 | 285 |
| Predictive | 93 | 93 |
| Counterfactual | 217 | 217 |

This is exact-string content deduplication, not semantic or script-template
deduplication. The source metadata has no participant or script-variation IDs;
the remaining candidate pool therefore cannot support participant-disjoint or
unseen-script claims. The reasoning mix is also strongly descriptive-heavy.

## Reproducibility artifacts

The original broad scan is retained externally at
`D:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-exposure-audit-20261111-attempt-01\exposure-report.json`
(SHA-256 `aad9ffc1e2d647c85ba0577914d9bd0609c694d3df90b3c2dc53fd2f313bc036`).
The corrected full re-scan, path classification, and content-overlap report is
retained at
`D:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-exposure-audit-20261111-attempt-05\exposure-report.json`
(SHA-256 `8da8b59847ddad68da09e029735a248661fb1d68ac95a8442aa2c10b452e076b`).
The latter preserves the full candidate ID list and per-ID evidence paths; the
large inventory is deliberately outside Git. The final 1,518-group content-
deduplicated candidate ID list has SHA-256
`af673f6282598366058d42df62b43a522d17d13766aca6e13e4d8a3a94c08382` when IDs
are sorted and joined with newline characters.

## Next gate

Use a separate branch to decide whether this candidate pool supports a
predeclared frozen-feature video Decision task. Before any fitting or scoring,
freeze a fixed group cohort, define text-only and video-only controls, validate
the media/hash and split gates, and document the missing participant/script
identifiers. Do not use these data to claim audio-video complementarity. If the
task design cannot address the descriptive-heavy mix and repeated script
templates, select another source rather than calling this a general video
Decision benchmark.
