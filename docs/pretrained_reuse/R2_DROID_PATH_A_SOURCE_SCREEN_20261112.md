# DROID video + robot-state source screen — 2026-11-12

## Decision

**Do not acquire the `droid_100` sample for a quality run.** The official sample
is a single 100-episode train split with no scene/session grouping in its
published TFDS feature schema. It cannot meet the roadmap's provisional
200-independent-scene validation floor even if every episode were an
independent group. No model, media, or annotation payload was downloaded or
opened, and no access condition was accepted.

The full DROID release remains a promising conditional source for a future
video + proprioception Decision task. The paper states that the dataset is
released under CC BY 4.0 and reports 76K demonstrations across 564 scenes, 52
buildings, and 50 collectors. The official docs describe three synchronized
camera views, low-dimensional state, and actions. However, the published
sample does not expose scene IDs, there is no official grouped train/validation
split, and the full release is too large to mirror locally. Do not treat the
paper-level scene count as an auditable split or the 100-episode sample as
independent validation.

## Pinned source and metadata-only evidence

- Official paper: <https://arxiv.org/abs/2403.12945>. It states that the full
  dataset is CC BY 4.0 and reports 76K trajectories, 564 scenes, 52 buildings,
  and 50 data collectors.
- Official data documentation snapshot:
  `droid-dataset/droid` commit
  `1548722f199772a43217c0d4c811c639be66e380`, file
  `docs/the-droid-dataset.md`.
- Official docs describe the RLDS release as 1.7 TB, the `droid_100` debug
  sample as about 2 GB, and the raw video release as about 8.7 TB. The docs
  say raw per-episode metadata includes building and collector IDs, while the
  TFDS sample's `episode_metadata` schema has only `file_path` and
  `recording_folderpath`.
- GCS object metadata for
  `gs://gresearch/robotics/droid_100/1.0.0/` listed 33 objects: 31 train
  shards and two metadata files. The publisher's `dataset_info.json` reports
  one `train` split, 100 total episodes, and 2,192,594,069 bytes. The checked
  `features.json` exposes 7D action vectors, robot state, three JPEG camera
  fields, and language instructions; it does not expose scene, building,
  collector, or validation-split fields.
- No source payload was fetched. Only the two small publisher schema files
  (760 and 18,665 bytes) and a 37,268-byte GCS object-metadata response were
  saved as provenance artifacts. Their SHA-256 hashes are recorded in the
  manifest and external receipt below.

External metadata-only receipt:
`D:/CodexArtifacts/pretrained-reuse-r2/droid-source-screen-20261112-attempt-01/`

| File | Bytes | SHA-256 |
|---|---:|---|
| `dataset_info.json` | 760 | `58646aa75fd054a0d8670edd676620267adb32550dc228545d70a3418abb1834` |
| `features.json` | 18,665 | `4c556e2e38e179e41089e07cf54273b1540d67d3c78a420898370329813c9a25` |
| `gcs-object-metadata.json` | 37,268 | `13728e1b75a484b06d6606c24138f189ec7c063dede32ccde643dd59168a9c5f` |

At screen time, D: had 1,918,322,425,856 bytes free. The full RLDS mirror is
not a safe local acquisition target at that capacity; the raw release is much
larger. The official sample's exact size fits, but its 100 episodes and lack
of grouping metadata do not justify a quality evaluation. No CUDA/GPU or
cloud compute was used.

## Decision-task fit and admission gates

A potential task is to select a typed next-action option from a short video
window and current robot state. DROID records continuous actions rather than
supplied multiple-choice labels, so an action-codebook/option construction
would have to be frozen using training data only. The current metadata does
not show that video and state are both required for that task.

Before considering selective acquisition, require all of the following:

1. Verify the official raw per-episode metadata fields and a stable grouping
   key. Build and hash a split with at least 200 independent held-out scenes
   where available, grouping conservatively by building and scene and auditing
   collector overlap.
2. Confirm the exact blur/privacy status of the selected media and that the
   CC BY 4.0 grant covers the selected videos, state, action, and annotations.
3. Freeze a supplied-option task and train-only action prototypes. Establish
   its label/noise semantics before loading validation labels into model work.
4. Prove task dependence with matched video-only, state-only, and joint
   baselines plus missing, shuffled, temporally offset, and option-order
   controls.
5. Use GCS object metadata to estimate and select only the required episode
   files. Do not mirror the full release. Stop if the grouped cohort, split,
   rights scope, or local resource estimate cannot be verified.

Until these gates pass, DROID is `REVIEW_NOT_ADMITTED` for fitting and quality
claims. This screen does not change the existing Path A models or datasets and
does not close R2-M. It identifies a plausible video + sensor source for the
next metadata-only feasibility step.

## Sources

- Official DROID paper: <https://arxiv.org/abs/2403.12945>
- Official dataset docs: <https://droid-dataset.github.io/droid/the-droid-dataset.html>
- Official data bucket: `gs://gresearch/robotics/droid_100/1.0.0/`
- Pinned docs repository commit:
  <https://github.com/droid-dataset/droid/commit/1548722f199772a43217c0d4c811c639be66e380>
