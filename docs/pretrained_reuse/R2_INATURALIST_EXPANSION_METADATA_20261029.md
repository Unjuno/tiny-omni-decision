# iNaturalist CC0 observer-disjoint expansion screen (2026-10-29)

## Result

This is a metadata-only coverage screen. It did not download image media, run
VideoPrism, fit a readout, select a checkpoint, or create a sealed audit.
The frozen candidate does not pass the predeclared minimum coverage gate, so
it is not admitted for training in this experiment.

The live iNaturalist API returned 16,000 observation rows across the same eight
bird taxa (2,000 per taxon). After excluding every observer, observation, and
photo identity present in the 2026-10-26 frozen cohort, 430 new train records
and 176 new validation records remained. These splits inherit the seed-17
observer split and do not reuse the old validation records. The expansion
metadata reports zero train/validation observer, observation, and photo
identity overlap. It does not establish normalized-media or perceptual-hash
disjointness because no media was fetched.

The minimum counts were fixed before filtering: 700 new train, 200 new
validation, at least 25 train and 10 validation records per taxon, with all
eight taxa represented. The overall count gate fails (430/700 and 176/200),
and Northern Cardinal has only 16 new train records. No threshold was relaxed
after observing the counts.

| Taxon | New train | New validation |
|---|---:|---:|
| Mallard | 53 | 23 |
| House Sparrow | 74 | 28 |
| Canada Goose | 54 | 18 |
| Rock Pigeon | 51 | 21 |
| American Robin | 53 | 24 |
| Great Egret | 80 | 31 |
| Great Blue Heron | 49 | 19 |
| Northern Cardinal | 16 | 12 |
| **Total** | **430** | **176** |

## Frozen metadata and provenance

The source query was Research Grade, photos present, and explicit CC0 at both
observation and photo level. The API is live rather than a dated source
snapshot. The source freeze records 3 cross-taxon photo IDs and 3 photo objects
without an eligible explicit CC0 license; those records were excluded before
the expansion filter. The filter then excluded 3,072 candidate rows belonging
to previously exposed observer groups. The final expansion contains no raw
observer IDs, names, coordinates, or descriptions.

Retrieval time: `2026-10-29T04:45:43Z`.

External files (not committed):

- Source: `C:\CodexArtifacts\inat-birds-cc0-expanded-source-v1-20261029\`
- Filtered expansion: `C:\CodexArtifacts\inat-birds-cc0-expansion-filtered-v1-20261029\`

SHA-256 values:

| File | SHA-256 |
|---|---|
| source `metadata.json` | `ef17f168fd209c864abf7da2473fc2e3a796f01a0763ca1d52ec611d7e716dbf` |
| source `train.jsonl` | `ab5647013857eb9d31e24208f19309e69d0ddfae20bb9bf8a947d70689d18758` |
| source `validation.jsonl` | `93591f853392a872cf69c186b6e69ad8ab75d720f2c26ca947c4ba02dcd9a02e` |
| filtered `train.jsonl` | `02aa68f414f28a2bb5a768bff431379d5f9c36936d71444f8408a0bd2024ee5f` |
| filtered `validation.jsonl` | `d5028e54f7d0000c0936f7319f6f8a598f6c458be02f10b2555ea7baf74729b4` |

The source files also hash the previously frozen 2026-10-26 manifests. The
filter implementation and its regression tests are in this branch. Focused
verification: 8 pytest tests passed; Ruff passed.

## Interpretation

This result shows that the 2,000-most-recent-observation query contributes
some unseen observer groups but does not provide the predeclared minimum for a
new coverage experiment. It says nothing about model accuracy or the source's
ultimate capacity. The existing 2026-10-26 readout results and all of their
artifacts remain unchanged. No image download or training followed this failed
metadata gate.
