# NVIDIA PhysicalAI SingleArm source screen — 2026-11-02

## Disposition

**`REVIEW_NOT_ADMITTED` for R2-M quality evaluation; promising candidate for a
bounded Video + proprioception action-choice pilot.** Rights,
storage, synchronized payload availability and a potentially useful action
label are supported by pinned publisher metadata and one verified training
episode. DINOv2-base supplies a rights-cleared per-frame CPU feature path. The
required task, independent scene grouping, temporal feature policy and
both-input necessity are not yet established. No training or task evaluation
occurred, and no GPU work was started.

## Pinned source and rights

- Publisher: NVIDIA, Hugging Face dataset
  [`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm).
- Dataset revision observed: `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- The publisher card declares **CC BY 4.0**, says the dataset is available for
  commercial use, and describes automatically generated Franka Panda motions
  in procedurally generated Isaac Sim scenes. Attribution remains required.
- The whole dataset is 15,138,436,544 bytes according to the publisher API.
  The `panda-open-cabinet-left` subset is 1,582,348,397 bytes, within current
  local free space; no full-subset download was made.
- A sorted SHA-256 inventory digest for the subset's 7,564 pinned file entries
  is `edf7843093cb499e0f6c91834bcb114cad74aee0438fae60b40d7e08ed0dd04f`.
  Digest recipe: UTF-8 encode sorted lines
  `rfilename<TAB>size<TAB>(LFS SHA-256 or Git blob ID)`, joined with LF and no
  final LF.

## Subset structure verified from pinned metadata

The official `meta/info.json` declares 1,512 episodes, 220,038 frames, 30 fps,
one task, and only a `train` split (`0:1512`). Its 25-dimensional observation
state contains end-effector pose, gripper state and joint position/velocity;
the listed state does **not** include scene object poses. The target action is
an 8-dimensional joint-delta/gripper vector. Each episode has two RGB streams
(world and hand cameras) and two depth streams at 512×512.

The subset's pinned media inventory is:

| Stream | Files | Bytes |
|---|---:|---:|
| World RGB | 1,512 | 734,010,701 |
| Hand RGB | 1,512 | 498,805,293 |
| World depth | 1,512 | 244,470,182 |
| Hand depth | 1,512 | 51,262,706 |
| Parquet episode records | 1,512 | 46,600,882 |
| Metadata | 4 | 7,198,633 |

Pinned metadata hashes:

- `meta/info.json`: SHA-256
  `d782c809c30490f96bb665343a39cf423626de8e638dd53905c379d1944ff6df`.
- `meta/tasks.jsonl`: SHA-256
  `1b56befc214182e48ccbc1a8b35d67d6be6a3bce92fa4ef054fb52377bb5b43d`; its
  only task is `panda-open-cabinet-left`.
- `meta/episodes.jsonl`: SHA-256
  `f81631538a9107e856e78c8bb040f58175e4720864727c4d192a0c5f191e775a`.
- `meta/episodes_stats.jsonl`: 7,077,948 bytes, SHA-256
  `e9e2f0019d14c2df8ea67cd88ca4a7a6b136aa2cabc425c15ee34be5ed160577`, and
  Git blob SHA-1 `d1589232f1a507e0f7e8f9f337b89ef0d4a6dea3`. Its 1,512 rows
  have unique episode indices and varying action statistics, but still carry
  no separate scene/layout identity. These summaries cannot prove that
  procedural scenes are independent across episodes.

## Local sample verification

At the exact revision above, one training episode was downloaded into the new
artifact directory
`C:/CodexArtifacts/pretrained-reuse-r2/physicalai-singlearm-source-smoke-01/`.
The Parquet file and the world/hand RGB videos each matched the publisher's
LFS SHA-256 and exact byte size. CPU PyArrow/PyAV inspection found 134 Parquet
rows and 134 decoded 512×512 frames in each RGB stream at 30 fps (4.4667 s).

The per-frame encoder smoke used Meta's
[`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) at
revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`. Its pinned safetensors
file is 346,345,912 bytes with SHA-256
`d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`; the
official DINOv2 repository explicitly licenses its code and weights Apache
2.0. Transformers 5.6.2 / PyTorch 2.6.0+cu124 loaded 86,580,480 parameters
(zero trainable) on CPU. Two synchronized RGB frames produced finite
`[2,257,768]` features in 0.444 s; a second load reproduced the output exactly
(maximum absolute delta 0). This verifies a per-frame image feature path, not a
temporal encoder or task result. Receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/dinov2-base-runtime-smoke-01/cpu-runtime-smoke.json`.

No depth media, other episodes, validation examples, sealed audit, or GPU was
loaded. The source receipts are `verified-sample-hashes.json` and
`verified-cpu-inspection.json` under the episode artifact directory.

## Task and integrity gates still open

The most plausible R2-M mapping is a **Video + proprioception action-choice**
task: at time *t*, select the next numeric action from supplied typed action
options, using the synchronized camera clip and robot state history. The
recorded 8-D action can provide supervision, but discretization/option
construction, horizons, sampling, baselines and acceptance criteria must be
fixed before fitting. The task is narrow: a single scripted cabinet-opening
instruction, not general language, image, audio or video understanding.

Specific blockers before admission:

1. The source supplies no official validation or audit split; all episodes are
   declared train. Create project-owned train/validation/sealed partitions
   only after verifying complete episode grouping.
2. `episode_index` is available, but a separate scene/layout identity and
   cross-episode scene-overlap evidence are absent from the inspected metadata.
   Episode-level splitting alone must not be described as scene-disjoint until
   this is checked.
3. Continuous action labels must be mapped to stable candidate actions without
   target leakage or a trivial candidate/option shortcut. Keep every frame and
   question from an episode in one partition.
4. Compare joint, video-only, proprioception-only, shuffled-state and
   time-shifted-state arms on the same frozen rows. Until the joint model beats
   the best singleton with grouped uncertainty and interventions, this is not
   evidence that both streams are necessary.
5. DINOv2-base now provides a rights-cleared local CPU per-frame feature path,
   but the time-aware feature policy and Decision runtime remain to be
   implemented. The V-JEPA 2.1 exact weight grant remains REVIEW; prior
   VideoPrism extraction has a Windows runtime-policy blocker. Do not silently
   substitute either model or claim that temporal features are already
   available.

The candidate is therefore **not yet admitted for fitting or R2 quality
claims**. Next safe work is a no-media episode/scene grouping and action-label
protocol audit, followed by a rights-cleared runtime feasibility check. The
source screen does not authorize access to any pre-existing sealed audit.
