# PhysicalAI SingleArm video + proprioception action pilot — 2026-11-02

## Disposition

This is a bounded wiring and feasibility pilot only. It does **not** admit the
source for R2-M quality evaluation, establish scene-independent generalization,
or justify product/runtime architecture changes. It tests whether frozen video
features and robot state can predict a coarse recorded demonstrator action in a
small episode-held-out development sample. The target is not an optimal or
successful action.

## Frozen source, partition, and selected sample

- Source: NVIDIA `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`, subset
  `panda-open-cabinet-left`, revision
  `73c00c18e7ca108e91846294b20a1d7cd5de9340`, publisher-declared CC BY 4.0.
- Split seed: 17; rank key is SHA-256 of dataset/revision/subset/seed/episode.
- The full inventory has 1,512 episode groups: 1,058 train, 227 development
  validation, 227 reserved. Every episode's world RGB SHA-256 is unique and
  there is no exact media-hash overlap between groups.
- Four previously inspected episodes (0, 1, 750, 1511) were forced into train.
  The pilot uses 64 other train episodes and 16 validation episodes, four
  windows each (256 train and 64 validation examples). The reserved partition
  was not downloaded or evaluated. It is public and not blind: source-level
  episode summary metadata was previously inspected. No scene/layout ID exists
  in metadata, so scene independence is UNKNOWN.
- Exact split membership and hashes are in
  `manifests/pretrained_reuse/physicalai-singlearm-cabinet-left-split-20261102.json`.
  Manifest SHA-256 is
  `52a845a2c2033347b105806b8458848156c7b0a9a7e7c6489fc02a10543dab90`;
  train/validation/reserved group hashes are respectively
  `414e69c329ebf1ded88e42272daa7e53634f02151bfd0841af33d062a8164f87`,
  `9979b76f68285eaee6e7dcdc3252a9bfa561286d8b8e7165313118f29fef1ddb`,
  and `6807cfbf64b565c77e906e137e7c1f2630cf1857ff34817bad3d98b4d31de913`.

## Task and controls

At each query, the model receives eight consecutive world-camera frames ending
at time *t*, plus eight corresponding 25-D robot-state rows from *t-7* through
*t*. Four positions are sampled at 20/40/60/80% of each episode's valid time
range. The target is the recorded 8-D action at *t+3* (100 ms at 30 Hz). The
Parquet timestamps matched `frame_index / 30` within 1e-4 seconds on every
selected episode, and decoded video frame rate was 30 Hz.

Eight action prototypes were fit only on train future actions after scaling
each dimension with train-only standard deviation. The prototype fit used
deterministic farthest-first initialization followed by Lloyd means. This is
coarse **quantized demonstrator-action prediction**, not policy optimality,
task success, or general robotics decision making. Class labels are prototype
IDs, with the complete fixed set shared by all examples.

The feature extractor was frozen DINOv2-base at revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`, CPU FP32. Linear softmax probes
used train-only feature normalization and LBFGS (maximum 200 iterations).
Compared arms use the same validation examples: proprioception history,
eight-frame mean visual feature, last-frame visual feature, and joint video +
proprioception. A video-episode-misaligned intervention swaps each validation
episode's video with the next validation episode's video while preserving
window order and proprioception. The option-only baseline is the train class
prior. No checkpoint selection or sealed/reserved evaluation was done.

## Results

| Validation arm | Accuracy | NLL | Brier | ECE | n |
|---|---:|---:|---:|---:|---:|
| Proprioception history | 0.7656 | 4.3195 | 0.4691 | 0.0021 | 64 |
| Video, mean of 8 frames | 0.7500 | 1.3050 | 0.4247 | 0.2097 | 64 |
| Video, last frame | 0.7656 | 1.5217 | 0.4362 | 0.2071 | 64 |
| Joint video + proprioception | 0.7969 | 1.5239 | 0.3823 | 0.1811 | 64 |
| Joint model, video replaced by train mean | 0.5938 | 1.9694 | 0.6633 | 0.2810 | 64 |
| Joint model, proprioception replaced by train mean | 0.7188 | 1.7101 | 0.5011 | 0.2430 | 64 |
| Joint model, episode-misaligned video | 0.4531 | 5.7025 | 0.9665 | 0.3681 | 64 |
| Train class-prior baseline | 0.2188 | 1.9026 | 0.8375 | 0.0117 | 64 |

The joint probe exceeds the best singleton accuracy by 3.1 percentage points
(0.7969 vs. 0.7656), but the episode-cluster paired bootstrap interval for the
joint-minus-proprioception difference is -9.38 to +15.63 points. For
joint-minus-8-frame-video it is -3.13 to +12.50 points. These 2,000 resamples
are over only 16 validation episodes (seed 17); the intervals are wide and do
not establish a reliable benefit. Episode-misaligned video lowers joint
accuracy to 0.4531, a 34.31-point paired drop (95% episode-cluster interval
20.31 to 48.44). Replacing video with its train mean drops Accuracy by 20.25
points (paired interval 6.25 to 34.38); replacing proprioception drops it by
7.90 points (interval 3.13 to 14.06). These fixed-model interventions suggest
both inputs affect this probe on these 16 episodes, but the small exposed
development sample, class-coverage gap and prototype target prevent a
generalizable complementarity or quality claim.

Validation contains no examples for action class 4 and only two for class 3.
The eight-class codebook therefore has inadequate class coverage for quality
claims. Calibration is also weak/inconsistent: visual and joint ECE are high,
while proprioception has very poor NLL despite high accuracy. The codebook's
train and validation normalized action RMSE are 0.4956 and 0.5688. Reported
validation metrics are exploratory and must not drive product selection.

## Runtime and artifacts

The 80 train/validation videos and matching Parquet files total 42,356,840
bytes. At the pinned publisher revision, all 160 files passed byte-size and
SHA-256 checks. Eight-way parallel transfer took 53.965 seconds. The three
earlier single-file transfer timings were 2.353, 1.644, and 1.678 seconds
(mean 1.892 seconds/file); do not use the unrelated iNaturalist rate for this
endpoint.

CPU feature extraction used PyTorch 2.6.0+cu124, Transformers 5.6.2, PyAV
18.1.0, PyArrow 25.0.1, Python 3.11.9, and DINOv2-base FP32. It processed 320
queries / 2,560 frame appearances in 549.122 seconds of encoder time and
23.825 seconds of video decoding. No GPU was used. Feature and result files
are stored outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2/physicalai-action-pilot-01/`:

- `download-receipt.json` — 160 source paths, bytes, hashes and timings.
- `feature-extraction.json` — query counts, timestamp checks, encoder time,
  pinned source/model revisions and feature hash.
- `features.npz` — frozen visual, state, action and grouping features.
- `pilot-metrics.json` — metrics, class counts, bootstrap, protocol and hashes.
- `validation-predictions.json` — all 64 IDs, targets, predictions and
  probability vectors for every probe/control.
- `probe-checkpoints.pt` — all fitted linear probe weights, train-only feature
  normalization, action normalization and action prototypes.
  Reloading each of the four fitted probe arms reproduced all 64 saved
  probability rows with maximum absolute delta 0.0.

Artifact SHA-256: download receipt
`ca8fb5ffa6ae51964cfe7f011c353890e2ce94d6726c518ea300b6fe32608617`, feature
archive `ebaace829f4e608746a8ffb657dfedcb8d865fae030d3f747a66f83047a1182a`,
metrics `655af288385601a84001f523c99da73d4c0d017c116d1e0dd3f49f6cfe943214`,
and extraction metadata
`37713ce533c46c8780fc83f86102e58f7b2003cd9a4f3a17e37012459df47eba`.
The original metrics hash above predates the addition of saved predictions and
checkpoints. The regenerated prediction and probe checkpoint hashes are
`FD496AB84C4EE9F9D943A1058457F44D2478738708A626DB353E3B43E91A7E13` and
`102965afe03de2cb6cb549e9e37df7355ea9063b605a086ee00079c3eb7c1da0`;
the final metrics file (including zero-delta checkpoint reload validation)
hash is
`335524FEDBCB8AFC973F33B1A842F662A3D125C804DBA858EA916102AFEB6EDD`.

Reproduction code is versioned at `scripts/build_physicalai_singlearm_split.py`,
`scripts/extract_physicalai_action_pilot.py`, and
`scripts/evaluate_physicalai_action_pilot.py`; split-builder regression tests
are in `tests/test_physicalai_singlearm_split.py`. Re-evaluation from the
saved features reproduced the metrics before writing per-example predictions
and probe checkpoints. Full feature extraction takes about ten minutes on the
measured CPU path.

## Next gate

Do not expand this pilot directly into a larger fit. First establish
episode-to-scene/layout identity or explicitly limit claims to held-out
episodes from this procedural source. Then pre-register better-covered action
targets, enough validation episodes, label/codebook stability checks, and
state/video-only and sensor-swap controls. A larger experiment must compare a
simple state-only baseline and frozen visual baseline under identical groups,
measure episode-cluster uncertainty, and demonstrate that both inputs provide
repeatable incremental value. Keep this source at `REVIEW_NOT_ADMITTED` until
those gates pass.
