# RoboPro action-probe sample preflight — 2026-11-23

**Status: data preflight passed; no model fit or quality evaluation yet.** This
report records a metadata-selected, bounded train/development sample for the
next frozen-feature Decision probe. RoboPro remains `REVIEW_NOT_ADMITTED` for
general multimodal quality because semantic action validity, physical scene
identity, and an official holdout are not established.

## Frozen selection

- Dataset: `Hoshipu/RoboPro`, publisher revision
  `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`.
- Existing project-created prompt-group split:
  `manifests/pretrained_reuse/robopro-prompt-development-split-20261123.json`.
- New selection: seed 17, SHA-256 ordering of exact prompt groups and episode
  IDs, 16 prompts per partition, four episodes per prompt. This yields 64 train
  and 64 development episodes (32 prompt groups total).
- The eight episodes and both prompt groups from the earlier runtime smoke were
  excluded to keep those inspected examples out of this sample.
- Sample manifest:
  `manifests/pretrained_reuse/robopro-action-probe-sample-20261123.json`,
  SHA-256 `eeab697bc4189ba2b4442ed53def960f248cf146b2cc947c0633905a41340686`.
- Selection reads metadata only and makes no choice from action labels,
  features, or validation scores.

## Download and integrity audit

Only the selected `cam_high` MP4 and Parquet files were downloaded, from the
official `lerobot/roboreal_all_80tasks/` subtree at the pinned revision. The
external receipt binds all 256 files to the sample manifest and records each
local SHA-256. Total acquired payload: 127,088,672 bytes (about 121.2 MiB).

The bounded CPU audit checked all 128 episodes:

- 14-D state and action arrays are finite; Parquet row/frame/episode/task IDs
  are contiguous and map to the frozen prompt.
- Every video decodes at 50 fps with its expected frame count.
- Maximum absolute Parquet timestamp to video PTS difference:
  `4.58e-7` seconds.
- Zero exact prompt overlap, MP4 SHA-256 overlap, or normalized 32x32 RGB frame
  hash overlap across the project train/development partitions.
- At 25%, 50%, and 75% relative episode progress, all four same-prompt action
  vectors were numerically distinct at L-infinity tolerance `1e-6` in all 96
  candidate sets.

The machine-readable audit is external at
`C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-02\data-audit.json`,
SHA-256 `3c4742411d895972dd2a3ec905db45133697516360723b1703555e3af23ab8ae`.
The download receipt SHA-256 is
`3c5da58198c80478526cde6513c29cd58954416cf19244a715bfeeec2c275ce3`.
Runtime was Python 3.11.9, PyArrow 25.0.1, PyAV 18.1.0 on Windows. No GPU or
cloud compute was used.

## Interpretation and next gate

This establishes a usable, synchronized small data slice and numerically
distinct action choices, not that each group has exactly one semantically
correct action. All source records are from publisher TRAIN; development was
created by exact prompt grouping and may share a physical lab, camera setup, or
unobserved parent task with training. A normalized frame non-overlap check
does not prove scene independence.

Next, extract frozen VideoPrism features only for these 128 selected episodes,
then compare a fixed small set of train-only readouts with text-only,
state-only, video-only, and joint inputs on the same development queries. Keep
results descriptive and grouped by prompt. Do not tune repeatedly on this
development sample or promote the source/model from this pilot.
