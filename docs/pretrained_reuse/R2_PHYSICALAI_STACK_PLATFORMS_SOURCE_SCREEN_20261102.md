# PhysicalAI SingleArm stack-platforms metadata screen — 2026-11-02

## Disposition

`REVIEW_NOT_ADMITTED` for model fitting and quality claims. This metadata-only
screen identifies a potentially stronger sensor+video task candidate than the
previous cabinet-left pilot: the state includes multiple object poses as well as
robot state, alongside synchronized video and recorded actions. It does not
prove scene-independent generalization, both-input necessity, decision quality,
or suitability for a product task. No media, Parquet episode payload, model
weights, or checkpoint was downloaded or evaluated in this screen.

## Pinned publisher evidence

- Dataset: NVIDIA
  [`PhysicalAI-Robotics-Manipulation-SingleArm`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm).
- Revision: `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- Subset: `panda-stack-platforms`.
- The pinned publisher `README.md` declares `cc-by-4.0`; its SHA-256 is
  `a30cfb2d0c40f39c054021ab79e6dabb03b57815394711eaf9c72dd0347a24a2`.
  Attribution is required.
- Pinned `panda-stack-platforms/meta/info.json`: 5,897 bytes, SHA-256
  `325ee4a7354291b57714a6980a8aecbc7f9a01d5ea66caf569cc82c8ad39c87a`.
- Pinned `panda-stack-platforms/meta/episodes.jsonl`: 1,313,432 bytes, SHA-256
  `d3d3a9f88df886d2fb005dbe4733d25020d1b0ac296046ea37eb59dda2752c73`.

The publisher metadata reports 17,629 episodes, 1,456,899 frames, and 30 fps.
Its only declared source split is `train: 0:17629`. The 81-dimensional state
contains four boxes' positions/quaternions and end-effector/gripper/joint
state; the recorded action is seven-dimensional. World and hand camera video
streams are listed. The upstream episode table contains 17,629 unique episode
indices, but its rows expose only `episode_index`, `length`, and `tasks`; no
scene, layout, environment, or procedural-generation identity is present.

## Research implication and gates

The state-rich subset may support a narrow task such as choosing among supplied
future-action candidates from recent video and object/robot state. Such a task
would be coarse demonstrator-action prediction, not task success or general
robotics competence. The source's episode IDs are the strongest exposed group
identity, but they cannot establish scene/layout disjointness. Before fitting:

1. Freeze the exact task, candidate construction, action target horizon, and
   candidate count without looking at validation outcomes.
2. Build a deterministic episode-grouped train/validation/reserved split from
   this subset's own pinned world-camera inventory, with exact media-hash and
   normalized-content checks. Do not reuse the cabinet-left split or assets.
3. Keep the reserved records out of normal iteration; describe the result as
   held-episode, same-source development evidence unless independent scene
   identity is established.
4. Use matched video-only, state-only, and joint controls plus state/video
   shuffles before claiming that both inputs help.
5. Measure the data footprint and available storage before downloading any
   media. This screen did not compute the full subset byte inventory.

Do not expand the prior cabinet-left fit based on its exposed validation
results. If the gates above cannot be met, keep this subset metadata-only and
retain the negative/uncertain cabinet-left pilot as recorded.

## Access and resource record

Only the pinned repository README, `meta/info.json`, and
`meta/episodes.jsonl` were fetched into process memory to inspect bytes,
SHA-256, schema, row count, and group fields. The 157 MB `episodes_stats.jsonl`
was not fetched. No train, validation, or reserved episode media were fetched;
no sealed audit was accessed. No GPU, model execution, fitting, cloud compute,
or paid resource was used.
