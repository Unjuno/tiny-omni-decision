# MAVD / Urbansas audiovisual source screen — 2026-10-21

## Decision

**Do not admit this candidate to training, validation, or audit yet.** MAVD-traffic itself is listed as CC BY 4.0, and Urbansas annotations are CC BY 4.0; TAU media bundled in Urbansas is noncommercial and is excluded. However, the source split cannot currently support a clean independent evaluation: the official MAVD train and validation annotation archives share parent recording sessions, while the additional Urbansas MAVD-labeled sites have not been mapped away from upstream test or unpublished source material. A provisional split was invalidated when source identity overlap became clear. No sealed audit split exists for this candidate.

## Source and license evidence

- MAVD-traffic official Zenodo record 4741232, version 0.1.1 / record revision 2: API metadata says `cc-by-4.0`; the source describes synchronized audio/video and event intervals. The DCASE catalog describes 47 files, 235 minutes, 3,427 event instances, and a supplied train/validation/test partition.
- Urbansas official Zenodo record 6658386, version 1.0 / record revision 2: annotation archive metadata and the included `LICENSES.txt` say annotations are CC BY 4.0. The same license file says the MAVD media subset is CC BY 4.0 and TAU media is noncommercial. Only MAVD annotations were used in this screen; TAU rows/media were excluded.
- Official files were limited to small annotation/metadata artifacts. **No audio or video media was downloaded or decoded.** No upstream test annotation archive was downloaded or opened.

Sources: [MAVD Zenodo record](https://zenodo.org/records/4741232), [DCASE MAVD catalog](https://dcase-repo.github.io/dcase_datalist/datasets/sounds/mavd.html), [Urbansas Zenodo record](https://zenodo.org/records/6658386).

## Split and label-integrity findings

The MAVD original TRAIN annotation archive lists 24 five-minute source files; VALIDATION lists 7. Both contain source-session prefixes `rivera0923`, `rivera1015`, and `rivera1016`. The official split therefore leaks at the parent recording-session level and cannot be reused as an independent split for this project.

The Urbansas annotations contain 540 MAVD-labeled 10-second clip identities and 15,639 video annotation rows across four location families. A filename-only hash split was briefly preflighted, assigning two locations to train, one to validation, and one to a candidate holdout. That preflight is **invalidated**: the holdout location, Rivera, overlaps source labels inspected separately in official MAVD train/validation files. Other Urbansas locations have not been demonstrated disjoint from upstream test or unpublished source material. Do not use the provisional split or its exploratory counts for fitting, selection, or audit.

A possible task label—same vehicle class annotated as audible and visible at a timestamp—is only class-level temporal co-occurrence. The annotations do not establish that a particular sound came from a particular visible track, so this source cannot support exact audio-to-object correspondence claims.

## Artifacts and execution

The external API snapshots, annotation files, license/readme files, and hashes are listed in `R2_MAVD_URBANSAS_SOURCE_SCREEN_20261021.json`; local copies are under `C:\CodexArtifacts\pretrained-reuse-r2\mavd-source-screen-20261021` and `C:\CodexArtifacts\pretrained-reuse-r2\urbansas-mavd-screen-20261021`. The source manifest records exact SHA-256 values.

No GPU, training, raw media, upstream test annotations, or sealed metrics were used. This source screen does not change the R0 task/device gates or support Architecture B/C work. The next candidate must have source-level provenance that excludes upstream test/unpublished records and enough independent recording groups for train and validation.
