# DCASE 2025 Stereo SELD source screen — 2026-10-20

## Decision

**Do not admit this release as a fresh R2 audio/video task or start a fit from
it.** The official DCASE 2025 task is a useful on/off-screen localization
benchmark, but its 30,000 five-second development clips are derived from the
same STARSS23 recordings already used in this repository's R2 STARSS23
development probe. Its on/off-screen labels are derived from the annotated
sound-source direction and the selected camera field of view; the documented
target does not establish that both audio and video pixels are necessary.
The DCASE page also does not specify a separate license for its processed
release. Keep the candidate at **REVIEW / NOT ADMITTED**.

This screen used official task metadata only. It did not download or open any
clips, labels, evaluation media, or sealed audit data. No model, corpus,
checkpoint, run, or Orbitune process was changed.

## Evidence from the official task page

- DCASE states the development set contains 30,000 five-second clips sampled
  and converted from STARSS23 recordings. Its evaluation set is another
  10,000 clips from both recording sites.
- The conversion turns FOA audio into stereo and equirectangular video into a
  perspective view. It rotates source directions into the selected view and
  derives the on/off-screen label from the direction and the field of view.
- Development clips are split by source room: two Sony and seven Tampere rooms
  for training; two Sony and five Tampere rooms for the development test split.
  The clips therefore inherit the STARSS23 parent recording and room identity.
- This repository already used STARSS23 parent recordings for the fixed
  train/calibration/validation partition in
  `R2_STARSS23_SPLITS_20261011.md` and measured the matched audio-only,
  video-only, and joint readouts in `R2_STARSS23_COMPARISON_20261011.md`.
  Repackaging or resampling windows from those recordings cannot be called
  fresh group-disjoint evidence. The DCASE evaluation set remains unopened.
- The official DCASE 2025 page explains the dataset derivation and challenge
  use, but does not state a distinct license grant for the processed clips and
  labels. The upstream STARSS23 terms do not by themselves establish the rights
  scope of every DCASE-converted product.

The source therefore misses two R2 admission requirements independently:
fresh parent-group isolation and truth-bearing evidence that requires both
streams. Resolving the processed-release license alone would not make it
eligible for the current joint-fit gate.

## R0 / compute status at the screen

The current live GPU observation is recorded separately from the earlier
06:58 UTC audit: `nvidia-smi` reported 15,821 / 16,384 MiB allocated, 15%
utilization, and CUDA PID 548. Process inspection tied PID 548 to Orbitune's
`compound_cfe_train`; supervisor PIDs 4696 and 23856 were alive. They were not
stopped, signaled, or modified.

For a future justified GPU job, the clean branch contains
`scripts/run_with_hotgpu_guard.py` and `scripts/hotgpu_preflight.py`. The
wrapper waits for all external CUDA PIDs to yield, reserves VRAM, rechecks for
a race, and only then launches the child. The child requires the matching
reservation metadata. This screen did not invoke the wrapper because this
candidate failed admission; no process was queued or scheduled.

PR #15 remains draft/open at `43974fb2f6b070e0182465edca12d061b65ed8dd`.
The deployment device and locked task battery/quality, calibration, latency,
energy, and time-cost thresholds remain open under R0.

## Reproduction and references

This was a read-only metadata comparison against the pinned internal STARSS23
split and result reports. The official source is:

- DCASE 2025 Stereo SELD task page: <https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content>
- STARSS23 source paper and release as linked by DCASE: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/e6c9671ed3b3106b71cafda3ba225c1a-Abstract-Datasets_and_Benchmarks.html>

No media, annotation, evaluation, or model payload was acquired. No GPU,
cloud compute, or cost was used. The branch preserves all existing artifacts.
