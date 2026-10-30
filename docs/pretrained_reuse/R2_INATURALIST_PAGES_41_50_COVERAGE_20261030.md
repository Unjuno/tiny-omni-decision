# iNaturalist CC0 pages 41–50 metadata coverage gate (2026-10-30)

## Result

The metadata-only extension does not pass the already-fixed coverage gate and
is not admitted for training. The source query returned 2,712 records across
the requested page window, all with explicit CC0 observation and photo
licenses. After excluding observer, observation, and photo identities present
in all earlier source windows and the fitted fresh-validation cohort, only 27
train and 14 validation records remained. They represent two of the eight
taxa (Mallard and Canada Goose); the other six taxa had no rows in this page
window. The predeclared minimum remains 700 train / 200 validation, at least
25 / 10 per class, and all eight classes represented.

| Taxon ID | New train | New validation |
|---:|---:|---:|
| 6930 (Mallard) | 16 | 10 |
| 7089 (Canada Goose) | 11 | 4 |
| **Total** | **27** | **14** |

The source freeze selected 658 observer/class groups before cross-window
exclusion; 617 were excluded as previously used observers. Final train and
validation observer, observation, and photo IDs are disjoint. No media was
downloaded, so exact-file or perceptual-hash checks were not possible. No
model features, metrics, or checkpoint were produced, and the sealed audit was
not materialized or accessed.

## Provenance

- Query: iNaturalist API `/v1/observations`; Research Grade, photos present,
  explicit CC0 observation and photo licenses; `date_added` descending;
  pages 41–50; seed 17; observer-hash 70/30 split.
- Retrieval time in source metadata: `2026-10-29T15:47:21Z`.
- Source directory, outside Git:
  `C:\CodexArtifacts\inat-birds-cc0-expanded-source-pages-41-50-20261030-attempt-01`
- Filtered directory, outside Git:
  `C:\CodexArtifacts\inat-birds-cc0-expansion-filtered-pages-41-50-20261030-attempt-01`
- Frozen source metadata SHA-256:
  `7a9908ff8dcc7c868bd7fbc49a2dacca3ed376184820a75a7ed250fafac05fca`
- Source train / validation SHA-256:
  `5072ad90d786ebe00cece7fa360789959cadcc0a48e1898be8a52e370d6d4570` /
  `b30d55598444cf9e7594bf66c697d97d472834a7b8b04d68c3a610b0bd5ef69d`
- Filtered train / validation SHA-256:
  `fd4ffc39a385987145385ce4fe9d946ed715d84a12c147e4f1d9d32650e160bd` /
  `d5a2f4b5e31c36ee9ee33ce5df4803df1740bfb59b2c4a34361a7cdefb30a636`

The query used the existing metadata-freezer and expansion-filter gates. The
worktree code and existing datasets/checkpoints were unchanged; this attempt
created only new JSON metadata/manifests under `C:\CodexArtifacts\`.

## Interpretation

The first 40 API pages already covered the useful new observer groups for six
of these taxa. The remaining query window cannot add a balanced eight-taxon
cohort at the declared minimum. This is a source-coverage result, not evidence
that the representation cannot improve. Do not relax the gate or train on the
27/14 rows. A future expansion needs a distinct eligible source or a new
predeclared task/split, with any new validation treated as development data.
