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
  sub-repositories. The public metadata reviewed here does not provide a
  consolidated, independently grouped release split with at least 200 groups
  per required evaluation partition.

## R2 gate assessment

| Gate | Evidence | Result |
|---|---|---|
| Rights for both synchronized modalities | Dataset cards display Apache-2.0; scope across every media/annotation component and person-related rights was not verified | Open |
| Relevant observations | Camera plus proprioceptive/force/tactile data are described | Promising, not sufficient |
| Independent grouped splits | No verified group-level split counts or group identity contract found | Fail closed |
| Supplied-option Decision task | Release describes manipulation demonstrations and policy learning; no fixed supplied-option decision labels were verified | Open |
| Both-input necessity | No paired unimodal-vs-joint task/evaluation evidence found | Open |
| Local feasibility | ModelScope search metadata reports a 51.04 GB collection; no payload was retrieved or storage feasibility tested | Not assessed |

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
