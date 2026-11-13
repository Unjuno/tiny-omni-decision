# RoboMIND 2.0 source/task screen — 2026-11-12

## Disposition

**Status: REVIEW_NOT_ADMITTED.** This is a metadata-only screen. No dataset
payload, media, annotations, or access-gated content was downloaded or opened;
no access agreement was accepted. Do not train or report quality from this
candidate until the R2 gates below are resolved.

## Official evidence

- The [official RoboMIND project site](https://x-humanoid-robomind.github.io/)
  describes RoboMIND 2.0 as more than 310,000 trajectories across six robot
  embodiments and 739 tasks, with camera observations and robot state. The
  [official RoboMIND 2.0 ModelScope card](https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0)
  displays Apache License 2.0 and describes camera, force, tactile, and
  proprioceptive streams. The [official Hugging Face RoboMIND 2.0 card](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND2.0)
  also displays Apache-2.0, but its repository currently contains no data
  payload. This does not establish the license scope for every source asset,
  annotation, or identifiable person in the complete ModelScope release.
- The [RoboMIND 2.0 paper](https://arxiv.org/abs/2512.24653) describes
  imitation-learning trajectories and simulated data. It does not specify
  independent-scene/session/collector train-validation group counts in the
  accessible abstract.
- The official ModelScope collection is split across embodiment-specific
  sub-repositories. A metadata-only clone of the top-level
  `X-Humanoid/RoboMIND2.0` repository was pinned at ModelScope commit
  `75a95b8ac4c4cdf8ea1e846d72761e6de92f199c`. Its README describes 310K
  trajectories, including 12K tactile-enriched sequences, and documents
  camera, robot-state, force, and tactile fields. The tree contains 1,894
  paths, but no top-level split manifest or independent-group counts. Large
  media/data formats are Git LFS tracked; the clone used
  `GIT_LFS_SKIP_SMUDGE=1` and `--no-checkout`, and no LFS objects were fetched.
  The actual per-subrepository/HDF5 group identities and split counts remain
  **UNKNOWN**. The older RoboMIND 1.x page shows a train/val directory example,
  but that example is not evidence about the 2.0 release.
- A further metadata-only check pinned the official simulation sub-repository
  `X-Humanoid/RoboMIND2.0-Tienkung-sim` at
  `a8e3d2e926cc286563bf9d641c23e3e658fe6edd`. Its default-template README
  declares Apache License 2.0 but gives no detailed dataset or split
  description. The `122-rotate_pot_handle/success_episodes` directory contains
  303 timestamped HDF5 episode pointers. The task path exposes no train/val
  partition. The IDs span six calendar dates, but no official metadata maps
  them to independent scene/session/operator groups; count 303 episodes, not
  303 independent groups.
  One LFS pointer reports a 13,768,050-byte HDF5 object; this is not a complete
  task-size estimate. No LFS object, HDF5, annotation, or media payload was
  fetched or opened.

## R2 gate assessment

| Gate | Evidence | Result |
|---|---|---|
| Rights for both synchronized modalities | Dataset cards display Apache-2.0; scope across every media/annotation component and person-related rights was not verified | Open |
| Relevant observations | Camera plus proprioceptive/force/tactile data are described | Promising, not sufficient |
| Independent grouped splits | One simulation task has 303 timestamped episodes across six dates but no declared train/val split or independent scene/session mapping; independent-group count is UNKNOWN | Fail closed |
| Supplied-option Decision task | Release describes manipulation demonstrations and policy learning; no fixed supplied-option decision labels were verified | Open |
| Both-input necessity | No paired unimodal-vs-joint task/evaluation evidence found | Open |
| Local feasibility | ModelScope reports 51.04 GB for the top-level collection and 194.21 GB for the Tienkung simulation sub-repository; one task's total pointer sizes remain UNKNOWN | Not assessed |

This candidate could support a future task such as choosing a typed next action
or predicting a failure state from synchronized camera and robot-state cues,
but that would require a predeclared label/option contract and evidence that
the task truly needs both streams. The current public metadata does not justify
that conversion or a model run. No new quality claim is made.

## Decision

Keep RoboMIND 2.0 outside the admitted corpus. Reconsider only after an
officially documented independent-group split, component-level rights scope,
and an option-based paired task with unimodal/missing/shuffled controls are
verified. Existing Path A checkpoints, corpora, and results remain unchanged.
