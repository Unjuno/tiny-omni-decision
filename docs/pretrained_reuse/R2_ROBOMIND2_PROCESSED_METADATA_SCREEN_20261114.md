# RoboMIND 2.0 processed release metadata follow-up — 2026-11-14

## Disposition

Keep `OneWorld-AI/RoboMIND2.0-processed` `REVIEW_NOT_ADMITTED` for Path A
quality fitting or evaluation. The pinned derivative is a potentially useful
video-plus-robot-state source, but its released train/validation assignment
contains exact encoded-video duplicates across splits, independent scene or
session groups are not documented, and no frozen supplied-option Decision
target is provided. Do not use its current validation split as an independent
quality estimate.

## Pinned source and inspected material

- Repository: `OneWorld-AI/RoboMIND2.0-processed` at
  `99b62f054a4118f1323483b9a78eac697e7d2e85` (verified with the Hub metadata
  API and `git ls-remote`).
- The pinned card advertises Apache-2.0. It says the derivative retains the
  Apache-2.0 terms of its three named upstream RoboMIND 2.0 releases. The
  upstream ModelScope cards also display Apache-2.0. This is a viable declared
  license path for further review, not a blanket clearance of identifiable
  people or every source asset.
- Only the pinned `README.md`, repository file metadata, `LICENSE` metadata,
  and `audit/duplicate_groups.jsonl` were inspected. No media, TAR/HDF5 member,
  action/state array, model, or sealed audit was opened or downloaded. The
  duplicate-group JSONL was read in memory only; its SHA-256 is
  `578aace2229cbf0a47a7c91eaddf57684102ed1cc2584985d692833bce7153e4` and its
  size is 13,000 bytes.

## What the source offers

The card describes 134,537 processed episodes (121,101 train and 13,436
validation) and approximately 846 hours. Each TAR episode has synchronized
video and per-frame state/action arrays plus task instructions in JSON
metadata. The source material is described as successful demonstrations; no
failure/outcome label or supplied-option Decision protocol was found in the
card. A next-action choice task would therefore need an explicitly defined
option construction and defensible labels, rather than treating continuous
demonstrations as ready-made Decision examples.

## Split and integrity findings

The published duplicate audit contains 48 duplicate groups. Eleven groups are
marked `cross_split=true`, covering 12 duplicate episode IDs. All 11 of those
groups are also marked `encoded_video_also_identical=true`. Thus exact video
content crosses the published train/validation boundary. Episode counts cannot
be treated as independent validation groups, and the published validation
split fails the media-identity disjointness gate.

The card exposes episode-level train/validation paths and an exact-payload
duplicate audit, but does not establish independent scene, collection-session,
or task-generation groups. It does not document a group-disjoint split. The
provisional 200 independent-group gate is therefore `UNKNOWN`, not passed.

The card reports a 2.6 TB total release. This is not itself a reason to reject a
small selected subset, but no subset acquisition or local runtime estimate was
attempted because the task/split gates fail first.

## Decision and reconsideration gate

This is a promising temporal visual-plus-state candidate, not yet a valid
Decision benchmark. Reconsider only after all of the following are established
without opening a held-out audit:

1. a task whose correct option and label are well-defined from the released
   observations, including a valid counterfactual/negative option policy;
2. an immutable train/calibration/development split grouped by underlying
   recording or scene, with zero video-content overlap and independent-group
   counts reported;
3. task/source leakage checks for instructions, task IDs, embodiment, and
   repeated demonstrations;
4. a selected subset with measured local storage and preprocessing cost.

Until those pass, do not download media, extract features, fit a readout, or
evaluate a model on this release. Existing Path A runs and source artifacts
remain unchanged.

## Primary references

- [Pinned derivative dataset card](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/tree/99b62f054a4118f1323483b9a78eac697e7d2e85)
- [Pinned duplicate audit metadata](https://huggingface.co/datasets/OneWorld-AI/RoboMIND2.0-processed/blob/99b62f054a4118f1323483b9a78eac697e7d2e85/audit/duplicate_groups.jsonl)
- [Official RoboMIND 2.0 source card](https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0)
- [Official Tienkung simulation source card](https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Tienkung-sim)
