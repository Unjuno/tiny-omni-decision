# iNaturalist CC0 bird metadata freeze (2026-10-26)

## Result

The predeclared eight-taxon metadata cohort was fetched and frozen outside Git.
This is an input-data preparation result only: **no media bytes were downloaded,
no image model was run, no checkpoint was selected, and no sealed audit was
materialized or opened**. The cohort is still not admitted as a product-quality
result; media identity, perceptual-duplicate, label-content, and visual
grounding checks remain open.

The live API returned 8,000 observation rows (1,000 per taxon). One photo ID
appeared under two taxa and was excluded; one photo object did not have an
explicit CC0 license and was excluded. The remaining eligible rows were
deduplicated by photo ID and capped at one selected observation per observer
per taxon. The output contains 2,325 rows: 1,642 train and 683 validation.
All selected records retain both observation-CC0 and photo-CC0 values.

## Frozen artifact

External directory (not committed):
`C:\CodexArtifacts\inat-birds-cc0-v1-20261026\`.

| File | Rows / bytes | SHA-256 |
|---|---:|---|
| `train.jsonl` | 1,642 / 595,812 bytes | `2c2f914b8575111836ea53cd2143b8621703a410d67c7c39a8ebc34781c66b2a` |
| `validation.jsonl` | 683 / 251,223 bytes | `9a40230bdfb2312701b62471adb999cb557d17c631a208f95d4d36c924ae0487` |
| `metadata.json` | 3,097 bytes | `05d95ecc54f3c5ea7c45158549d7f5f8b10a9147c048e8dc51cc9658c5b117c4` |

Metadata retrieval time: `2026-10-25T21:30:57Z`. Query parameters were
Research Grade, photos present, observation license CC0, photo license CC0,
`date_added` descending, 200 rows per page, five pages per species. This is a
live API response, not an immutable source snapshot; live totals and frozen
manifest hashes are stored in `metadata.json`.

| Taxon | ID | Train | Validation |
|---|---:|---:|---:|
| Mallard | 6930 | 202 | 85 |
| House Sparrow | 13858 | 257 | 116 |
| Canada Goose | 7089 | 232 | 100 |
| Rock Pigeon | 3017 | 222 | 86 |
| American Robin | 12727 | 204 | 74 |
| Great Egret | 144455 | 233 | 104 |
| Great Blue Heron | 4956 | 179 | 75 |
| Northern Cardinal | 9083 | 113 | 43 |

The split is a deterministic seed-17 70/30 observer-group split. The frozen
files contain 729 distinct train observer groups and 304 validation groups;
observer-group SHA-256, observation ID, and photo ID overlaps are all zero.
There are 2,325 unique observations and 2,325 unique selected photo IDs.
The earlier feasibility-only 70/15/15 counts are not this frozen split; no
audit role was assigned here. Coordinates, descriptions, observer identifiers,
and names were not retained. The source photo URL and source IDs exist only in
the private external manifests required for acquisition and provenance.

The first fetch attempt was interrupted after a long request with no progress
output and produced no artifact. A single 200-row page was measured at about
10 seconds. Progress logging was added, and the unchanged query completed on
the second attempt. No live supervisor or GPU process was stopped or changed.

## Next gate

Before fitting a readout, download only the frozen train and validation rows
to a new external cache, strip image metadata on normalized copies, verify
HTTPS host and byte limits, and hash original and normalized pixels. Check
exact and perceptual media overlap across splits, broken images, and whether
the selected image actually depicts its observation taxon. Then extract
features with the existing pinned VideoPrism-B CPU-only worker. Compare the
trained readout against balanced and shuffled-label baselines and image
permutation/zero-image controls. Report the narrow eight-class accuracy as a
development result only. This cohort has no sealed audit and cannot substantiate
the 95–98% multi-task/multimodal target.

The existing Open Images probe is not a substitute: its image-ablation control
left every prediction unchanged. The new experiment must pass an image
dependence control before its classification score is interpreted as visual
understanding.
