# iNaturalist ascending-page CC0 expansion gate (2026-10-31)

## Decision

This metadata-only extension fails the previously frozen expansion gate and is
not admitted for training or evaluation. After excluding observer, observation,
and photo identities from every prior iNaturalist cohort through pages 41–50,
only 20 records remain: 15 train and 5 validation records, all Mallards. The
other seven taxa contribute zero new records. The predeclared minimum is 700
train / 200 validation records, at least 25 / 10 per taxon, and all eight taxa.
No images were downloaded, no features or predictions were produced, and the
existing scorer was not evaluated.

## Frozen query and identity audit

The official iNaturalist observations API was queried for the same eight taxa
as the existing diagnostic: research-grade observations with photos, explicit
CC0 observation and photo licenses, ordered by `date_added` ascending, pages
1–5 at 200 rows per page (40 requests / 8,000 returned observations). Retrieved
metadata was frozen at `2026-10-30T20:04:03Z`. Raw API rows, observer IDs/names,
coordinates, and descriptions were not retained. The existing fixed source
freezer's `seed17|observer_id` SHA-256 recipe was verified against a previously
used observation before cross-window filtering.

The new source freeze selected 1,813 one-observation-per-observer/taxon groups
(1,276 train / 537 validation), including all eight taxa. Cross-window filtering
then excluded 1,793 records because their observers were already present in the
prior cohorts. Only the 20 Mallard rows remained. Train/validation observer,
observation, and photo IDs are disjoint in the retained set. This does not
replace an eventual file-level/perceptual duplicate check; no media were
downloaded because the frozen minimum-count gate already failed.

Three preflight API requests took 2.86 s, 1.94 s, and 2.04 s (mean 2.279 s),
which projected 91.2 s for 40 requests. The actual sequential query took about
10 minutes from runner creation to frozen source receipt. The projection was
not reliable for this API window; future estimates should use the full-window
runtime, not these three samples. The successful response counts and complete
query are in the external metadata receipt.

## Provenance and hashes

- Prior source-builder worktree: `codex/r2-inat-expansion-roadmap-20261029`,
  commit `e3909424a583c7fc5a216e7f277624b6403569d3` (read-only, clean).
- Original freezer SHA-256:
  `496f6578ec641b1f73bae2d762ebcdb51c49ac3d9966b2f7690a94c0f04da1fc`.
- Expansion-filter SHA-256:
  `8f49664d08c414e6595a803d64dd632c2afbb561e59d50baaaf31af8fd1ff0dc`.
- Executed freezer copy changes only `order=desc` to `order=asc`; its SHA-256 is
  `8a149f2e75509de660ba7b3b66095d631d0b78d5bcc4c808c9b45dff9c2f86c4`.
- Source train / validation SHA-256:
  `356e081be2cb9a9d4e8789270bcd2132a37272f15ced79095e2d02eaadfc4214` /
  `7c910a7bc9f679eacdb10b5f39af7e1b7adea9d2b8780721e5c882c715a80f54`.
- Filtered train / validation SHA-256:
  `af631c52118c7e260ab3209185e3256238bb9d69f580c2dbaac1cc08c0afa953` /
  `88ec51a0a0d57c0bc1b1a7ced28524c57f4ee441c9ab1a011239008108558e4d`.
- Source metadata SHA-256:
  `d04876002ceed25c3a4845dfb7fcf77a6b6e163165217c547bbfad70abf76216`.
- Filtered metadata SHA-256:
  `2c9b7cc98f65e5aed8c5e8346b684e6ebc074521ea79cb92708ef0056606b197`.
- Source and filtered receipts remain outside Git under
  `C:\CodexArtifacts\inat-birds-cc0-ascending-attempt-01\source` and
  `C:\CodexArtifacts\inat-birds-cc0-ascending-attempt-01\filtered`.

## Implication

Do not download the 20 remaining images or reuse this cohort for model
selection. The existing eight-taxon image validation cannot be expanded with
this API window while retaining the existing observer-disjoint contract. A
future image-quality experiment needs a different admitted source or a
predeclared task with enough new independent groups. This result says nothing
about the model's general image capability and does not open the sealed audit.
