# CATER / Clevr-4 numeric source-ID collision screen — 2026-11-13

## Result

This read-only metadata audit excludes CATER rows whose numeric scene/video
index collides with any previously used Clevr-4 image ID. It creates no new
corpus and does not admit CATER: different-ID visual-content duplicates and
the exact grant attached to the hosted pre-generated archive remain
unverified. No video member, Clevr-4 sealed audit, model, feature, or label
outside the two CATER training lists and the non-sealed Clevr-4 manifests was
read. No GPU or training was used.

## Inputs and reproducibility

- CATER source commit: `13a19643f1a2fb24e931df25abd74353e4f2fdcb`.
- CATER `train_subsetT.txt`: SHA-256
  `62b1733632a7f447bcc1fcb1a4e1cb4472fabf14d756fe5cabd5df4afddd23cb`.
- CATER `train_subsetV.txt`: SHA-256
  `e320cfe0af43d8c1256885096952a4a0bb4e31475332f720a6b530cccd425b81`.
- Clevr-4 source/configuration and encoder are pinned in the frozen input
  receipt from `path-a-image-videomae-clevr4-100k-seed17`.
- Only these non-sealed Clevr-4 media manifests were read:
  - train, 72,010 rows; SHA-256
    `e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410`;
  - selector, 1,000 rows; SHA-256
    `4c551e91d4e02bd896380476d0e59f1348a01ed3f95a5eed7e0d39cb9a311dee`;
  - one-time reporting, 5,296 rows; SHA-256
    `0dd87a1303cb25bbc87ed3d69a2acde5ecef02f27bec16985297e9d685b09525`.
- The sealed audit was not loaded. The exact command is the checked-in
  `scripts/audit_cater_clevr4_id_collisions.py` invocation with the two list
  paths, the three manifest paths above, and output
  `D:\CodexArtifacts\pretrained-reuse-r2\cater-source-screen-20261113-attempt-01\clevr4-index-collision-screen.json`.
- Machine-readable result SHA-256:
  `94bf68dc1a56fea3f227a3552a22f6d8e49ecd76b916dd3c4069f91a2b2b6f50`.

The audit parsed CATER's six-digit filename suffix and upstream label, then
compared the suffix against the numeric suffix in every Clevr-4 `asset_id`.
This is a conservative source-ID collision rule: a match is excluded even if
the underlying media differ. It is not a pixel or scene-content duplicate
test. The earlier CATER metadata audit verified that each selected scene's
`image_index` matches its archive-member suffix and found no normalized scene
duplicates between the two CATER lists.

## Counts after conservative exclusion

| CATER partition | Original rows | Same numeric ID as used Clevr-4 | Kept | Kept examples per 64-way class |
|---|---:|---:|---:|---:|
| `train_subsetT` | 3,079 | 1,400 | 1,679 | 16–42 |
| `train_subsetV` selector | 770 | 331 | 439 | 2–13 |

All 64 labels remain represented in both filtered partitions. There are 78,306
unique numeric IDs in the three non-sealed Clevr-4 manifests combined. None of
the Clevr-4 selector or one-time reporting IDs collided with the CATER lists;
all detected collisions were with the Clevr-4 training manifest. The filtered
sets retain zero shared CATER ID between `train_subsetT` and `train_subsetV`.

The selector's minimum support of two clips per class is too small for a
strong class-wise quality claim. If later gates pass, report it as a
development-only selector with class counts and uncertainty, not as an
independent audit.

## Rights and identity gates remain explicit

The pinned [CATER README](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/README.md)
states that CATER is Apache-2.0 licensed. The pinned
[video generator](https://github.com/rohitgirdhar/CATER/blob/13a19643f1a2fb24e931df25abd74353e4f2fdcb/generate/render_videos.py)
records CC BY 4.0 as its default generated-data license, and the reused CLEVR
assets are covered by the official [CLEVR CC BY 4.0 notice](https://cs.stanford.edu/people/jcjohns/clevr/).
Keep both attribution notices if the candidate is used. The archived
per-scene JSONs do not carry a per-archive license receipt, so this audit does
not infer that every hosted file used the generator default.

The CATER member ID namespace is `CATER_new_*`, while its scene metadata also
contains a `CLEVR_new_*` filename field. Removing identical numeric suffixes
prevents direct ID reuse with the Clevr-4 assets already used here, but cannot
rule out identical rendered content under different suffixes. No video pixels
were acquired or compared. Therefore CATER remains
`REVIEW_NOT_ADMITTED` pending archive-scope confirmation and a bounded,
non-sealed content-identity check. The official validation/test lists remain
unopened.

## Verification

- Focused CPU tests: `3 passed`.
- Ruff on the new audit and tests: passed.
- No source dataset, model, checkpoint, or prior result was modified.
