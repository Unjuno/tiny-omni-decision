# iNaturalist bird metadata preflight

This is a bounded metadata screen, not an admitted dataset or model result.
On 2026-10-28 JST the official API was queried for eight bird names already
listed in the [prior source screen](R2_INATURALIST_BIRD_IMAGE_SOURCE_SCREEN_20261026.md).
Each request selected research-grade observations with photos, observation and
photo license filters CC0/CC-BY, ascending observation ID and at most 50 rows.
Requests were spaced by at least 1.1 seconds after each response. Only aggregate
counts and response hashes were saved; observer IDs, coordinates, individual
observation rows and media were not retained.

For this conservative screen a row was counted only when the returned taxon
name exactly matched the requested species, the observation license was CC0 or
CC-BY, and at least one non-hidden photo had CC0 or CC-BY with attribution.

| Species | Returned | Exact-name eligible | Distinct observers |
|---|---:|---:|---:|
| Anas platyrhynchos | 50 | 44 | 13 |
| Passer domesticus | 50 | 50 | 13 |
| Branta canadensis | 50 | 50 | 15 |
| Columba livia | 50 | 1 | 1 |
| Turdus migratorius | 50 | 50 | 17 |
| Ardea alba | 50 | 49 | 16 |
| Ardea herodias | 50 | 50 | 16 |
| Cardinalis cardinalis | 50 | 42 | 8 |

There were 37 distinct eligible observers across all eight samples; 22 appeared
in multiple species. Their overlap means species-local splits are insufficient.
Among returned rows, 49/50 `Columba livia` results were labeled as the domestic
subspecies; the `Cardinalis cardinalis` search returned eight differently named
taxa. Therefore name filtering alone must not be treated as an exact taxon-ID
contract. The exact-name rule is a conservative diagnostic, not a claim that
subspecies are wrong or that the complete source lacks usable examples.

This earliest-ID sample is nonrandom and the live API is not an immutable
export. Page/photo metadata is not independent ownership verification, and the
filter does not necessarily license every photo in a returned observation.
The source requires separate observation and photo license checks; consult the
[official API](https://www.inaturalist.org/api) and
[license guidance](https://help.inaturalist.org/en/support/solutions/articles/151000173511-how-do-licenses-work-on-inaturalist-should-i-change-my-licenses-).
No media, split, training, validation, or quality evaluation was created.
Before admission, pin an export and taxonomy IDs, define descendant handling,
verify individual photo attribution/licenses, and split globally by observer.

The aggregate output is
`C:/CodexArtifacts/pretrained-reuse-r2/inaturalist-bird-metadata-preflight-20261028-attempt-01.json`,
SHA-256 `a49b7b156e3234ff06cda66e20671cd8b108e8caae451ec88c7613f7b03dd28a`.
It includes the per-response hashes but not the response bodies, so the live
API data cannot be reconstructed from this receipt alone.

## Follow-up: explicit taxon IDs

Each of the eight names returned exactly one active species-level taxon from
the current official taxa API. Their IDs, in the table order above, are 6930,
13858, 7089, 3017, 12727, 144455, 4956 and 9083. An additional request per
ID with the same earliest-50 and license filters returned no rows outside the
queried species' ancestry. The `Cardinalis cardinalis` ID query returned 50
exact species records, unlike the name query which also returned other taxa.

Taxon-ID queries include descendants: 49 of 50 `Columba livia` records were
descendants, as were six `Anas platyrhynchos` and one `Ardea alba`. All 400
returned rows had at least one photo carrying a CC0/CC-BY field and attribution,
and an observer ID. Those fields are metadata declarations, not ownership
verification; other photos on a row can have different terms. There were 43
distinct observers, 24 occurring in multiple queried taxa, and no duplicate
observation IDs within these returned samples. Per-taxon unique observer counts
were 15, 13, 15, 19, 17, 17, 16 and 8. The source-order and privacy limitations
above still apply. No labels were assigned to descendants for a corpus: an
explicit and immutable parent/descendant mapping is still required.

Taxon-resolution receipt SHA-256:
`c9ddc29f658ebcc7c0e355d2b043b27c6f819e728cab9d0761407df90390b538`.
Taxon-ID sample receipt SHA-256:
`20bbfeb59164d9a77fa9ae5bceff4720a82a24a33283b757003384e31d93efea`.
Both are in `C:/CodexArtifacts/pretrained-reuse-r2/` with prefixes
`inaturalist-bird-taxon-ids-20261028-attempt-01` and
`inaturalist-bird-taxon-preflight-20261028-attempt-01` respectively. They are
aggregate receipts, not saved API responses, a frozen export or a training set.

## Bounded development snapshot attempt

The new `scripts/snapshot_inat_birds.py` fixed the eight taxon IDs, queried two
100-row pages per taxon, selected a licensed attributed photo per observation,
and assigned train/validation using a stable observer-derived key. It preserves
descendant taxon IDs while assigning the requested parent ID. All 16 HTTP
requests returned 200, and each returned 100 rows that passed the local metadata
gate. The final global uniqueness check raised `ValueError: duplicate observation
or photo`; the command exited 1 before creating its output directory. It did
not retain the response rows or identify whether the duplicate was an observation
or a photo, so the precise cause is unknown. No corpus, media or model resulted.

Attempted command, with `PYTHONPATH=src;.`:
`python -u -m scripts.snapshot_inat_birds --output-dir C:/CodexArtifacts/pretrained-reuse-r2/inat-bird-development-snapshot-20261028-attempt-01`.
Source commit `b593aaa6`; three focused tests and Ruff passed before the attempt.
Do not silently discard duplicates or reuse this failed attempt as a corpus.
A future retry must first distinguish collision types and define deterministic
handling, then recheck global observer/photo disjointness and label conflicts.
Observer keys are pseudonyms, not anonymous data. The snapshot would remain
development-only and require per-photo rights, media and visual-duplicate checks.
