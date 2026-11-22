# RoboAug video + proprioception source/task screen — 2026-11-13

## Disposition

**Promising candidate; not admitted for fitting yet.** The pinned official
dataset card and repository README state Apache-2.0 for the dataset. The
source contains synchronized robot camera frames and proprioceptive/action
state. One train episode was downloaded to an external artifact directory
and schema-checked; no model, feature cache, corpus, train/validation split,
or quality result was created. The remaining task-group and task-construction
gates below must pass before fitting.

## Pinned source and metadata-only inventory

- Dataset: [X-Humanoid/RoboAug-Datasets](https://huggingface.co/datasets/X-Humanoid/RoboAug-Datasets)
- Hub revision observed from the dataset API: `08622781145a3791e8f7f26978ca162689da0c66`.
- Code/release repository: [Open-X-Humanoid/RoboAug](https://github.com/Open-X-Humanoid/RoboAug),
  pinned `main` commit observed with `git ls-remote`:
  `3bd6ac54d9b0c6ad5a35ad1298f8f09545495631`.
- The official dataset README describes three robot platforms, per-episode
  HDF5 trajectories, RGB/depth camera observations, robot proprioception and,
  for UR5e, a natural-language task instruction. It states that no train/val
  split is shipped and every episode occurs once.
- The UR5e `ur_move_lemon_from_plate_to_bowl` task has 295 episodes / 46,766
  frames in the README. The pinned Hub tree reports 295 HDF5 files totaling
  60,630,486,696 bytes. Every episode ID is dated `0313`; observed IDs span
  `0313_145704`–`0313_181251`. Date/session independence is therefore not
  established by the source metadata. The task has enough episode IDs for a
  provisional episode-group split, but no official or independently
  documented session grouping.

## Bounded sample acquisition and schema check

Exactly one train episode was downloaded outside Git to:
`D:\CodexArtifacts\pretrained-reuse-r2\roboaug-action-selection-preflight-20261113-attempt-01\`.

- Path: `Single_Arm_UR_5e/ur_move_lemon_from_plate_to_bowl/success_episodes/0313_153402/data/trajectory.hdf5`.
- Size: 130,634,632 bytes; SHA-256:
  `3597f4456749e824a0bc33e101146d21a6715fe6fc0fe273bb00bcffb4b878b4`.
- Transfer: 21.49 seconds, approximately 6.08 MB/s on this single-file sample.
- HDF5: 102 aligned frames; four RGB camera arrays and four depth arrays;
  `puppet/arm_joint_position` `(102, 6)`, `puppet/hand_joint_position`
  `(102, 1)`, and `puppet/end_effector` `(102, 6)`. Numeric state values
  checked in this episode are finite. Front/left/wrist RGB streams have 102
  distinct encoded frames; top camera has 64.
- Instruction: `move lemon from plate to bowl`.
- All 102 `is_intervene` values are zero. The field's relationship to the
  README's “teleoperated demonstrations” description is unresolved; it will
  not be used as a label or filter without clarification.
- The end-effector's 6D values include angular components; observed 6.25-rad
  95th-percentile step norm suggests angle-wrap risk. Any candidate action
  target must use a defined representation (e.g. position deltas plus wrapped
  angular deltas), not raw Euclidean differences over all six coordinates.
- A manual view of one temporary front-camera frame showed robot arms and a
  work surface, with no visible person or identifying detail. This is only a
  single-frame privacy screen, not a whole-corpus privacy finding; the
  temporary PNG preview was removed immediately after inspection.

The machine receipt and schema summary are outside Git at the directory above;
the schema summary SHA-256 is
`f804466908adad127a8b7a377b5a88d4f53c042bc4257b6533f757d2f27d3933`.
The one sample is not evidence of full-task integrity or quality.

## Frozen VideoMAE runtime smoke

The exact real sample was passed through the locally pinned VideoMAE V2
ViT-Small K710 checkpoint after strict state-dict loading. On the RTX 3080
Laptop (Torch `2.6.0+cu124`, CUDA `12.4`, FP32), 16 evenly spaced front-camera
frames produced a finite `(1, 384)` feature in `0.6182` seconds. The encoder
has `22,153,286` parameters and reported `108,215,808` CUDA allocated bytes
after forward. Feature SHA-256:
`3bdc9661e3c8838b64fd344ae25af0ed35907ae29f206efc760d567397df2b9d`.
Runtime receipt SHA-256:
`74cb397732fda63564962d0a51fbf86c3af8a19a516fb902407bcc227825f2b9`.

This is a one-episode feature-extraction smoke, not a fit or quality result.
The Orbitune training PID seen before the smoke disappeared during the
bounded forward; no process was signalled or killed, and the GPU returned to
11 MiB / 0% after the smoke. Treat that as observed self-yield/exit timing,
not proof of why the separate process exited.

## Candidate Decision task and gates

A bounded candidate is **next-motion selection**: given instruction, recent
video frames, and current robot state, score a fixed set of candidate next
end-effector motions and return option-ID probabilities. This is a
project-created action-choice task, not a supplied benchmark label. Before
training it needs:

1. a deterministic episode-group train / validation / untouched evaluation
   split, with episode and time-window isolation; the same-date collection
   context and effective number of independent groups must be reported;
2. a train-only definition of candidate motion options and hard negatives,
   avoiding future-frame, future-state, or target-derived option leakage;
3. label and task-only, video-only, proprio-only, and video+proprio controls,
   including episode-paired input shuffles, to show whether both streams help;
4. confirmation of `is_intervene` semantics and a broader small-sample privacy/
   scene-content review before extracting or retaining more frames;
5. measured storage/transfer budget. One task alone is about 60.6 GB; the
   21.49-second sample rate is not a reliable full-download estimate.

No additional episode was acquired, no GPU was used, and no training,
validation scoring, sealed-audit access, or product claim was made. If the
group/label/privacy gates fail, stop this candidate and continue R2 with the
next admissible source. This video+sensor candidate would not by itself
establish audio, broad video, or all-modality quality.

## Primary sources

- [Pinned RoboAug robot-dataset README](https://github.com/Open-X-Humanoid/RoboAug/blob/3bd6ac54d9b0c6ad5a35ad1298f8f09545495631/dataset/robot_dataset/README.md)
- [Pinned RoboAug dataset overview and license](https://github.com/Open-X-Humanoid/RoboAug/blob/3bd6ac54d9b0c6ad5a35ad1298f8f09545495631/dataset/README.md)
- [Pinned Hugging Face dataset card](https://huggingface.co/datasets/X-Humanoid/RoboAug-Datasets/tree/08622781145a3791e8f7f26978ca162689da0c66)
