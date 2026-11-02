# iNaturalist CC0 expansion metadata follow-up (2026-10-29)

## Result

The original pages 1–10 expansion did not meet the predeclared training-coverage
gate. Two subsequent metadata-only query windows extended the search without
changing that threshold or opening the sealed audit. Each window covered API
pages 11–20 or 21–30 for the same eight taxa, seed 17 and observer-group split.
No image media was downloaded and no model was trained or evaluated.

After sequentially excluding every observer, observation and photo identity in
the original 2026-10-26 cohort, the pages 1–10 expansion, and the earlier
retained records from pages 11–20, the two later windows contribute:

| Source page window | Train | Validation | Northern Cardinal train / validation |
|---|---:|---:|---:|
| 11–20 | 503 | 264 | 15 / 11 |
| 21–30 | 418 | 182 | 34 / 14 |
| **Combined** | **921** | **446** | **49 / 25** |

The combined frozen metadata manifest passes the predeclared count gate:
at least 700 train, 200 validation, 25 train and 10 validation per taxon, and
all eight taxa represented. It reports zero train/validation overlap for
observer, observation and photo identities. A separate replay confirmed both
split file hashes and all six cross-split identity checks. The first two
windows individually fail the gate; only their sequentially filtered union
passes. Do not combine unfiltered source pages or alter the recorded split.

## Source and provenance

The API is live rather than a dated snapshot. Queries were bounded to pages
11–20 and 21–30 per taxon (within the API's first 10,000-result window), with
Research Grade observations and explicit CC0 observation/photo licenses. The
second source window returned 15,759 manifest-eligible observations before
observer-group splitting; five observations lacked an eligible photo and
seven photo objects lacked explicit CC0. No sealed audit was materialized.
Manifests retain only pseudonymous observer-group hashes, observation/photo
IDs, labels, licenses and source URLs; names, raw observer IDs, coordinates and
descriptions are not retained.

Frozen external artifacts (not committed):

- Pages 11–20 source: `C:\CodexArtifacts\inat-birds-cc0-expanded-source-pages-11-20-20261029-attempt-01\`
- Pages 21–30 source: `C:\CodexArtifacts\inat-birds-cc0-expanded-source-pages-21-30-20261029-attempt-01\`
- Combined filtered expansion: `C:\CodexArtifacts\inat-birds-cc0-expansion-filtered-pages-11-30-20261029-attempt-02\`

Combined filtered hashes:

| File | SHA-256 |
|---|---|
| `metadata.json` | `8e1c8e65c281018657caa3edc8c2f0374d934b34a59fe949deba9baa9088ba22` |
| `train.jsonl` | `362a562568910676a15b71d42233fbeb8f6811116afcb6ef22f846bcf83e8618` |
| `validation.jsonl` | `c3f846c677516b5f08c82f6ea12a39146bf9a890be66306985f202622d6ddcc0` |

Each metadata record binds all previous/source metadata and manifest hashes.
The first and second source metadata hashes are respectively
`e999f95d0bfd80dcb4c5ee1d93f5f9b793677251415022aa8cac9a611e8695d3` and
`591f939115710c3cf31ecd9be003599b8d20005b60f849914302459768a75bed`.

## Decision and limits

This passes only the metadata coverage and identity gates. It does not pass
media acquisition, normalized-image/perceptual deduplication, or training
readiness. At the time of this check C: had 0.53 GiB free and the separate
Orbitune training process was still resident on the RTX 3080 Laptop GPU
(15,821/16,384 MiB reported used). No Tiny Omni GPU work was started. Preserve
the new metadata artifacts; acquire media only after the disk reserve can be
restored and the exact download/storage plan is measured.

This result does not change the prior one-time 176-record transfer evaluation,
does not make its validation eligible for further model tuning, and says
nothing about image quality, broad Decision quality, or four-modality product
quality. The original failed expansion report remains intact; this document
records the later metadata-only follow-up.
