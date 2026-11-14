# TartanDrive 2.0 metadata-only R2 gate — 2026-11-15

## Disposition

Keep TartanDrive 2.0 `REVIEW_NOT_ADMITTED` for a product-quality Path A
Decision evaluation under the repository's provisional gate. Its published
multimodal sensors are relevant, but the pinned metadata inventory exposes at
most 79 recording bags, and the paper says there are no explicit supervised
classification labels beyond sparse annotations. The available evidence does
not establish 1,000 task examples across 200 independent groups or a frozen
supplied-option Decision target. Do not acquire media or start feature/model
work on this candidate for the current acceptance task.

## Pinned evidence

- Toolkit repository: `castacks/tartan_drive_2.0`
- Source commit: `fe893b9b4132bfbd30ba84a839872b3a855e8176`
- Inventory: `assets/files.yaml`, Git blob `bfe2dceb0e56164156e3737a0bebed9ec8d5ab5d`, 297,703 bytes.
- The inventory parses to 79 distinct top-level `bags/.../` directories and
  79 distinct timestamp strings. These are recording-directory counts, not
  independent participant/session counts. Since group identity is not resolved
  for each run, 79 is only an upper bound on bag-level groups.
- The official project page describes seven hours of camera, LiDAR, inertial,
  GPS and proprioceptive data, and states CC BY 4.0 for the dataset; the toolkit
  is MIT. The license screen is positive but does not supply labels or split
  independence.
- The paper says all sequences were human teleoperated and records driver ID,
  date/time, context, weather, lighting, course/location and run-level sensor
  metadata. Co-pilot annotations include unusual events and a weak 1–5 score
  only in some runs. The paper explicitly states there are no supervised
  classification labels outside those annotations.
- The source paper also describes potential grouped splits by driver, time,
  speed and GPS region. No per-run independent-group count or compatible
  option-selection benchmark was established from the metadata inventory.

## Work boundary

This was a source-metadata inspection only. No bag, media, annotation payload,
model checkpoint, feature, prediction, GPU or evaluation split was accessed.
No previously observed benchmark or legacy artifact was changed. The candidate
may be reconsidered only if a task-level source audit proves adequate
independent groups and a defensible Decision target; counting timesteps or
frames as independent examples is not acceptable.

## Sources

- [Official TartanDrive 2.0 project page](https://theairlab.org/TartanDrive2/)
- [Pinned upstream inventory file](https://github.com/castacks/tartan_drive_2.0/blob/fe893b9b4132bfbd30ba84a839872b3a855e8176/assets/files.yaml)
- [TartanDrive 2.0 paper](https://arxiv.org/abs/2402.01913)
