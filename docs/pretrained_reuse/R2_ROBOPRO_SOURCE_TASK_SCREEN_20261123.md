# RoboPro source and task screen — 2026-11-23

## Disposition

**Status: `REVIEW_NOT_ADMITTED` for model fitting or quality evaluation.**
RoboPro is a promising, rights-cleared source candidate for a narrow text +
video + proprioception action-choice task. It is not yet an admitted R2-M
benchmark: its public metadata supplies demonstrations, not independently
validated correct/incorrect action choices or a scene-held-out split. Do not
read this screen as evidence of Decision quality.

## Pinned source and rights

- Dataset: [`Hoshipu/RoboPro`](https://huggingface.co/datasets/Hoshipu/RoboPro)
- Revision: `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0` (Croissant release
  v1.1.0; pinned 2026-11-23).
- The dataset card and Croissant metadata declare **CC BY 4.0**. The Croissant
  record states this is human teleoperation of a robot, has no human faces,
  voices, biometric data, or other personal information, and describes the
  released recordings as synchronized RGB plus robot state/action. These are
  publisher declarations; media was not inspected.
- The Hub currently reports a repository size of about 43.5 GB. No full clone
  or model was downloaded. The initial screen fetched only the three metadata
  files below; a later integrity smoke fetched one 13.7-KB train Parquet and one
  607.7-KB `cam_high` MP4. No other episode or camera was fetched.

| Pinned metadata | Bytes | SHA-256 |
|---|---:|---|
| `meta/info.json` | 3,235 | `91c7d161a416685b08f9db5b195a6ba02f7047d966fa188e0262f854af386cfe` |
| `meta/tasks.jsonl` | 110,465 | `8e114f65fd235eeac8fcd1b32e3c95a859613c6e4af119179366394b976d65bc` |
| `meta/episodes.jsonl` | 1,443,285 | `88492784987d483618b5144f37bac8c3036bdaaa11d2df730f7b26a804b68658` |

## Metadata findings

The pinned `info.json` declares LeRobot v2.1, 16,000 episodes, 3,728,773
50-Hz frames, 48,000 videos from three RGB cameras, 1,622 task descriptions,
and 14-dimensional `observation.state` and `action` vectors. The videos have no
audio stream. Its only split is `train: 0:16000`; there is no publisher-provided
validation, test, or audit split.

The 16,000 episode metadata rows each map exactly to one of the 1,622 task
prompt IDs. Across those prompt IDs, episode counts have minimum 2, median 9,
and maximum 79; 815 prompts have fewer than 10 episodes and none has 200.
These counts do not mean there are only 1,622 episode groups. They show that
prompt-level held-out estimates would be sparse, and the card's 80 underlying
base tasks cannot be recovered from the inspected metadata as an explicit
base-task/scene key. The card describes clean and cluttered variants and warns
that scenes are fixed per task variant and that held-out scenes are needed to
test scene generalization.

The publisher declares episode 15,999 to be a byte-for-byte duplicate of
episode 12,000 because an original source HDF5 was unrecoverable. Metadata
independently confirms both have the same prompt and length (328 frames), but
their media/action bytes were not fetched or hash-compared. Keep this pair in
the same future split or exclude the duplicate from scoring; do not count both
as independent evidence.

## One-episode schema and clock smoke

To verify that the advertised action-choice ingredients can be joined, one
official TRAIN episode was fetched at the same pinned revision:

- Parquet `episode_000000.parquet`: 13,656 bytes,
  SHA-256 `490285071affe235e500e17563d827b0871ecb6719c060e35fa10830c9906063`.
- `cam_high/episode_000000.mp4`: 607,681 bytes,
  SHA-256 `caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198`.
- The Parquet has 169 rows, matching episode metadata length; state and action
  each decode to finite 14-D float vectors. `frame_index` and `index` are
  contiguous from 0, `episode_index` stays 0, and `task_index` 8 maps to the
  episode's `reposition the bottle` prompt.
- Parquet timestamps run from 0.0 to 3.359999895 s at a median 0.0199999809-s
  interval. The H.264 640x480 video decodes at 50 fps into 169 frames with PTS
  from 0.0 to 3.36 s. Maximum absolute Parquet-timestamp/video-PTS difference
  is `1.14440918e-7` s for this camera and episode.
- All row-count, dimension, finiteness, index, task mapping, 50-Hz, video decode,
  and PTS-alignment checks passed. Runtime: Python 3.11.9, PyArrow 25.0.1,
  PyAV 18.1.0. Machine-readable output and both payload hashes are in
  `C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-schema-smoke-20261123-attempt-01\robopro-schema-smoke.json`
  (SHA-256 `2c7a2b675c822ba2e4da1508fca250ab65b59c8139794be9276899292aa6040f`).

This verifies one episode's table/video clock mapping only. It does not establish
dataset-wide synchronization, semantic task correctness, action validity,
scene independence, or Decision quality. No pixels were displayed or used for
feature extraction; no model ran.

## Candidate Decision task and unresolved labels

A plausible task is: given the task instruction, current RGB observation, and
14-D proprioceptive state, rank a small set of candidate next-action commands
or short command snippets and return probabilities as JSON. The logged
14-dimensional action at 50 Hz can provide a behavioral target. This would be
**demonstrator-action prediction**, not task-success prediction: the source
does not label success, failure, safety, or whether alternative actions are
also valid. A candidate set made from one logged action plus arbitrary action
negatives could create false negatives. Freeze the action horizon, option
construction, target ambiguity policy, and no-model priors before fitting.

The source is suitable for testing whether video contributes information
beyond joint state and task text, but that is not established by its metadata.
The first protocol must compare task-only, state-only, video-only, and joint
models on identical episode groups, then include missing, shuffled, stale, and
time-offset controls. Video/state synchronization must be verified from actual
Parquet timestamps and decoded frame PTS before any temporal claim.

## Admission gates before any model run

1. Pin and audit the official task-to-prompt/variant mapping. If a base-task or
   scene key is absent, state precisely whether the proposed result is only
   held-out-episode generalization within seen tasks and a fixed lab scene.
2. Freeze a behavioral action-choice contract that avoids claiming success
   labels, specifies multi-valid-action ambiguity, and makes the options
   available at inference time.
3. Create a deterministic episode-grouped development split; exclude or
   co-locate the declared duplicate. Hash every assignment and check all
   selected file references before fetching media.
4. Predeclare the joint-versus-best-unimodal comparison, calibration metrics,
   episode-cluster intervals, temporal controls, and model-selection rule.
5. Estimate selected-subset bytes and local decode/feature costs. The full
   repository is about 43.5 GB; do not acquire it wholesale. Keep data and all
   run artifacts on C: and preserve the existing 2-GiB free-space reserve.

The one-episode smoke above verifies table/video clock alignment as an
admission preflight, not as a quality experiment. Only after the remaining gates
pass should a bounded frozen-feature experiment be considered. No model
experiment was started. Existing Teacher, Path A, BERT, Orbitune, data,
checkpoints, and open PRs were left unchanged.

## Execution record

- Screen date: 2026-11-23 (JST).
- Local metadata directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-metadata-screen-20261123-attempt-01`.
- Local schema/clock smoke directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-schema-smoke-20261123-attempt-01`.
- Download method: HTTPS `curl.exe --fail --location` against the pinned
  Hugging Face `resolve/<revision>/.../meta/` URLs; all three files were
  SHA-256 hashed after receipt.
- Only the one pinned TRAIN Parquet and matching `cam_high` video described
  above were read. No other media, pretrained model, model output, official
  test, or sealed audit was read.
- No GPU/model training or feature extraction was started; the RTX 3080 was
  occupied by an unrelated Orbitune process at the live check.
- Machine-readable inventory:
  [`robopro-source-task-screen-20261123.json`](../../manifests/pretrained_reuse/robopro-source-task-screen-20261123.json).

## Sources

- [Pinned dataset card](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/README.md)
- [Pinned Croissant metadata](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/croissant.json)
- [Pinned `info.json`](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/lerobot/roboreal_all_80tasks/meta/info.json)
- [Pinned `tasks.jsonl`](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/lerobot/roboreal_all_80tasks/meta/tasks.jsonl)
- [Pinned `episodes.jsonl`](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/lerobot/roboreal_all_80tasks/meta/episodes.jsonl)
