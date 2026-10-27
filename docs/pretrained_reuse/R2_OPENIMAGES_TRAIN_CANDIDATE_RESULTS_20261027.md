# In-scope train image candidates and page declarations

The [fixed protocol](R2_OPENIMAGES_TRAIN_CANDIDATE_PROTOCOL_20261027.md)
selected 297 candidate images from the verified train label and image metadata
objects. All are distinct by ImageID, original MD5 and normalized Flickr
author identity; none has an ID, original MD5 or author identity present in
the pinned official validation image metadata. All previously used 176 train
image IDs were excluded. These are metadata checks and do not detect all visual
duplicates or author aliases.

| Target | Candidates | Matching Flickr page declaration |
|---|---:|---:|
| Toy | 27 | 23 |
| Door | 30 | 23 |
| Food | 13 | 13 |
| Plant | 30 | 26 |
| Sculpture | 17 | 15 |
| Vehicle | 30 | 25 |
| Clothing | 30 | 27 |
| Furniture | 30 | 28 |
| Building | 30 | 27 |
| Window | 30 | 25 |
| Animal | 30 | 26 |

The existing page checker found 258/297 declarations matching the recorded
license and author. Other outcomes were 31 HTTP 404, three HTTP 403, two
landing-page mismatches and three license mismatches. Page matching does not
authenticate ownership or make an item automatically admissible. No media
was downloaded, no model run, and no validation labels or test data accessed.

This is a fixed candidate inventory, not a role-balanced training set. Exact
four-choice construction, alternative role counts, option-only controls,
individual ownership, media bytes and source/content overlap remain open.
The shortfalls were not filled or hidden after observing the result.

Private artifact directory:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-in-scope-train-candidates-20261027-attempt-01/`.
Raw SHA-256: `candidates.json`
`67a9b9a468b868c82ec43cc73b33fe477fb8b97fb079f1035a2c67e051a07d5a`;
`private-attribution.jsonl`
`e9db942ff0f81162586391f70789945d048749a5570d8686e5b60b8c8b924a46`;
`private-page-checks.jsonl`
`5f2b299dde5646711ab0a785643904aa0dd993f3bcd710cd9b76eadd568db9fc`;
`page-check-summary.json`
`e28568057f60501eb77b90c657df0d12e72641516c3b36309b880f5aff7693a5`.
The existing checker's internal `checked_receipts_sha256` hashes its LF text
payload; on Windows this differs from the raw on-disk CRLF byte hash above.
Candidate and page receipts include private source attribution, so are not
committed. Both source objects and local input hashes are pinned in the
candidate result.
