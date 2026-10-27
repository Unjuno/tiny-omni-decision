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
