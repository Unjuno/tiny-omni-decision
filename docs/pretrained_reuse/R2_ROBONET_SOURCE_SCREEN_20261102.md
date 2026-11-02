# RoboNet source and task screen — 2026-11-02

**Disposition: `REVIEW_NOT_ADMITTED`.** Metadata only; no archive, sampler,
media, features, labels, model weights, or evaluation split was downloaded or
opened. No model or training run was started.

## Pinned publisher evidence

- Publisher repository: [`SudeepDasari/RoboNet`](https://github.com/SudeepDasari/RoboNet),
  default branch `master`, commit `d83eee20f39653c3f8e7c349df7350e8a9e9f7a7`
  (commit date 2020-01-13).
- At that commit, `LICENSE_DATASET` is the Creative Commons Attribution 4.0
  International Public License (Git blob `2c9703f8892627eeaa195e3abee3b11f5656f767`);
  the repository separately marks its code MIT. This is a publisher claim for
  the licensed material and does not establish rights over any third-party
  content beyond what the licensor controls.
- The project wiki describes image, state, and action sequences, and gives a
  36 GB full archive and an approximately 100 MB starter sampler. The project
  page reports 15 million frames, 113 camera viewpoints, seven robot platforms,
  and contributions from four institutions. The starter sampler's coverage and
  independent-group count are unknown.
- The wiki's example loader exposes image/state/action tensors and a `train` /
  `val` interface, but the reviewed instructions do not define a frozen
  source-session or scene-group split, nor prove those partitions are disjoint.

## R2/R0 gates

RoboNet has no audio stream, which is acceptable for the roadmap's optional
sensor path, but that path still needs a Decision task where video and state
provide complementary evidence. The publisher material reviewed here describes
robot video prediction and inverse models, not a fixed-option task with a
verified both-input requirement. A recorded action can also expose a shortcut
if options are derived directly from state/action labels. No such protocol or
unimodal/missing/misalignment controls are established by this source screen.

The reported 162,000 trajectories are not 162,000 independently verified
evaluation groups. The inspected publisher metadata does not establish
participant, collection-session, or scene identities, so the roadmap's minimum
of 200 independent evaluation groups remains unproven. The project instructions
describe one large archive; its full 36 GB size exceeds the current local C:
free-space observation of 1,402,617,856 bytes. The 100 MB starter sampler may
be small enough to store, but cannot be assumed to meet the frozen sample and
group minimums. Nothing was downloaded to test it.

**Decision:** do not use RoboNet for fitting or quality evaluation yet. Reopen
only after publisher metadata establishes asset and independent-group keys and
a train/validation/sealed split, and a fixed-option video-plus-state task passes
shortcut review. Do not use the sampler as evidence of the full dataset's
coverage. No sealed audit exists for this screen, and no audit data were
created or loaded.

## Provenance

- Pinned code/tree commit: `d83eee20f39653c3f8e7c349df7350e8a9e9f7a7`.
- Dataset-license Git blob: `2c9703f8892627eeaa195e3abee3b11f5656f767`.
- Source: [publisher repository](https://github.com/SudeepDasari/RoboNet),
  [publisher wiki overview](https://www.robonet.wiki/), and
  [publisher download/split instructions](https://github.com/SudeepDasari/RoboNet/wiki/Getting-Started).
- Local GPU use: none. Cloud use/cost: none. Payload bytes: zero.
