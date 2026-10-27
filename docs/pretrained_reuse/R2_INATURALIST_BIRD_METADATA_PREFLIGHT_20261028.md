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

Attempt 02 used commit `6ecc5185`, which distinguishes duplicate observation
and photo IDs and adds a regression test. All 16 responses again returned 200;
the final check failed specifically with `ValueError: duplicate photo`. No
output directory or corpus was created. The prior attempt did not preserve rows,
so this identifies the second attempt's collision, not necessarily the first's.
The number, labels, owners and cross-split status of collisions remain unknown.
The check still fails closed; no deduplication or replacement was performed.

Attempt 03 was an aggregate-only diagnostic using commit `7c945eac`, with the
same eight taxa and two 100-row pages each. Of 1,600 rows, zero observation IDs
were repeated and two photo IDs appeared twice each. Both photo groups had
different parent species labels; neither crossed the proposed observer split,
had multiple observers, or had conflicting declared photo attribution/license.
These are results of a new live API read, not a reconstruction of attempt 02.
A photo may depict multiple species; the observations alone do not establish a
single correct class. Do not silently choose one label or remove these examples
from an already fixed evaluation. No training corpus, images or row-level
identities were saved. The aggregate diagnostic receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/inat-bird-collision-diagnostic-20261028-attempt-01/summary.json`,
SHA-256 `720453ea4f01f1adfeba4a4fd08a75d21ac75679865e930b3b0ea1f0e0586152`.
Five focused tests and Ruff passed. Before creating a single-label development
dataset, define and record a quarantine policy for ambiguous photo groups;
future evaluation must retain and explicitly report the quarantine coverage.

### Development snapshot policy, fixed before attempt 04

Keep all rows associated with any repeated photo ID in a separate private
quarantine file; do not choose among their labels, and do not use them for
training or validation. Retain their split, labels and photo metadata, hash the
quarantine and report its size, split and class coverage. Duplicate observation
IDs still fail closed. Remaining rows must have globally unique photo IDs and
observer-disjoint train/validation assignments. This is an exploratory metadata
snapshot, not a fixed evaluation set: any later benchmark must report quarantine
coverage and assess multi-object labels rather than retroactively remove hard
evaluation examples. Photo hashes, near-duplicates, ownership, attribution
verification and independent observation sessions remain separate gates.

### Attempt 04 outcome

Commit `7e3fc66b` captured 1,600 metadata rows under the fixed policy. Two
duplicate photo groups (four rows) were saved in quarantine; both contain
different parent-species labels. The remaining 1,596 candidates comprise
1,356 train / 240 development-validation rows. Candidate and quarantine hashes
were independently recomputed. No observer key crosses the two candidate splits.

The train portion has 90 distinct observer keys, but its largest observer owns
481 / 1,356 rows (35.5%); development validation has 21 keys, with its largest
owning 74 / 240 rows (30.8%). These earliest-ID, nonrandom samples have strong
observer concentration and do not satisfy the provisional independent-group
screen. Species-local counts, quarantine counts and response hashes are in the
receipt. Observer hashes are pseudonyms and the private files include attribution
and media URLs; they are not committed to Git. No media was acquired and no
model was trained. Do not admit this snapshot for model quality evaluation.

Directory:
`C:/CodexArtifacts/pretrained-reuse-r2/inat-bird-development-snapshot-20261028-attempt-04/`.
SHA-256 `summary.json`:
`2cf53577929fe16f6d111bc4d6716edda9c59161354fab3af92522053f2f59b2`;
`private-snapshot.json`:
`1d97f5368855c80abdd98bfb95cdf0c6359b21bef7c8ab580318f3b3c914801d`;
`private-quarantine.json`:
`a83ddfe6369a46f2525d94173e2bb4ced63b52d1bd4f12e242cad8a993cac3e3`.
The source response hashes identify what was received but the full responses
were not retained; the sanitized snapshot is fixed, not a frozen source export.

### Observer versus photographer limitation and official export

The API snapshot's `observer_key` derives from the observation's user ID only.
The [official open-data metadata schema](https://github.com/inaturalist/inaturalist-open-data/blob/main/Metadata/README.md)
separately identifies the observation observer and the photo's photographer.
The API snapshot did not capture an independently verified photographer ID.
Therefore zero observation-observer overlap **does not establish photographer
disjointness**, and the split is not admitted. Names embedded in attribution
are not a reliable substitute for identity. The source table exposes separate
observation, photo and taxonomy metadata; individual photo licenses still need
checking independently of observation licenses.

On 2026-10-28 JST a read-only HTTPS HEAD of the official monthly S3 metadata
archive returned status 200, length 35,966,549,880 bytes, last modified
2026-09-27 17:38:16 UTC, and ETag `8d2b2b78d0c3d7dd2a3457e2e996f74b-69`.
The `-69` multipart ETag is not a SHA-256 or evidence of full-content integrity;
the `latest` URL is mutable and no version ID was supplied. Nothing was
downloaded. The [provider's dataset description](https://registry.opendata.aws/inaturalist-open-data/)
states that photo licenses vary and metadata is updated monthly. A future
acquisition must pin a specific object, verify bytes and source consistency,
and budget storage for the archive and streaming processing before use. It
must group both observation and photographer identities, including transitive
connections. No existing candidate snapshot was modified.

The public bucket listing additionally returned a dated
`metadata/inaturalist-open-data-20260927.tar.gz` object. Its HEAD response had
the same length and multipart ETag as `latest`, with last-modified
2026-09-27 17:31:41 UTC. Matching metadata makes this a candidate for a pinned
download but does not verify full bytes or guarantee the object cannot change;
use conditional requests and compute a full SHA-256 if it is acquired. The
current ~33.5 GiB compressed size is not an estimate of decompressed storage or
processing RAM. No download or extracted source data exists from this check.

### Archive prefix and transfer encoding check

A conditional HTTPS byte-range request for the first 1,048,576 bytes of the
dated archive returned the expected 206 range, total length and multipart ETag.
The first request used `requests`' automatic content decoding and returned
2,573,142 decoded bytes instead of the requested compressed length; it was
rejected and saved nothing. A second request read the raw response with HTTP
content decoding disabled and returned exactly 1,048,576 compressed bytes,
SHA-256 `c06b96d059b69f11e649078ab8e80acc723dcfa2b9cb451a656f1789e267de04`.
The server reports `Content-Encoding: gzip` even when identity was requested.
Future download/hash code must count and hash raw object bytes, not transparent
HTTP-decompressed bytes, and verify content ranges and conditional ETag.

The bounded gzip/tar prefix shows a directory followed by `observations.csv`,
whose tar header declares 32,899,873,979 uncompressed bytes. No observation
rows were inspected or saved. This is only the first member's header, not a
complete archive listing, source SHA-256 or extraction-size measurement.
A streaming filter may avoid materializing this table, but full source integrity,
later member sizes, resource limits and the photo/observer join remain unproven.

An additional raw 16 MiB conditional range read took 8.266 seconds (1.94 MiB/s
including request setup); a linear extrapolation to the 35,966,549,880-byte
archive is approximately 17,720 seconds (4.9 hours), not a reliable ETA or a
guarantee of sustained transfer. The prefix SHA-256 was
`6e579c0cd96a3c65d467ed3ceb42f8f732680a3c76b01ea6377435051935ab0c`.
Nothing was saved from the range request. At this point C: had 77,900,357,632
bytes free; this is a transient measurement and excludes decompression costs.

Reading only the first CSV header revealed `observation_uuid`, `observer_id`,
`latitude`, `longitude`, `positional_accuracy`, `taxon_id`, `quality_grade`,
`observed_on`, and `anomaly_score`. It contains no observation-license field.
The plan requires CC0/CC-BY separately for observations and photos, so this
archive's observation table alone cannot establish the existing license gate.
Locations were not read or saved. Resolve a version-consistent license join or
use the source only for the narrower photo-metadata purpose before claiming
admission. No full-archive transfer has been started.
