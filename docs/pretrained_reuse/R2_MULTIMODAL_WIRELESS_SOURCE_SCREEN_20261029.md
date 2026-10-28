# Multimodal-Wireless source screen — 2026-10-29

## Disposition

`REVIEW_NOT_ADMITTED`. The source has a promising synchronized camera/sensor/
wireless-channel structure for a narrow temporal decision task. The official
project page states CC BY 4.0. However, the public source pages inspected do
not specify a supervised decision target, an independent train/validation/test
split, per-archive sizes, or checksums. No data were downloaded or opened, and
no model was trained or evaluated.

## Pinned source evidence

- The [official overview](https://le-liang.github.io/mmw/) describes 161,400
  frames across 4 virtual towns, 16 communication scenarios, and 3 weather
  conditions. It lists synchronized CSI, LiDAR, RGB/depth cameras, IMU, radar,
  GPS, and bounding-box data. The same page states the dataset is licensed
  under CC BY 4.0.
- The [official data-collection description](https://le-liang.github.io/mmw/data-collection.html)
  says the streams are aligned by six-digit frame number and documents a
  directory hierarchy of town / scenario / agent. It identifies camera, point
  cloud, metadata, and channel-path files, including multipath gains, delays,
  and AoA/AoD. This gives a candidate grouping key but does not prove a
  pre-existing split or an independent evaluation protocol.
- The [official scenario list](https://le-liang.github.io/mmw/scenarios.html)
  assigns 5 scenarios to Town03, 5 to Town05, 2 to Town07, and 4 to Town10.
  Grouping by town is possible in principle, but four towns are too few to
  assume a stable independent test without inspecting scenario overlap and
  fixing the split before feature extraction.
- The [official download page](https://le-liang.github.io/mmw/download.html)
  links a OneDrive folder and a mainland mirror. It states that the data were
  updated on 2025-11-17 and the channel data were augmented on 2025-11-23, but
  does not list total or per-scenario bytes or file checksums. The official
  [code repository](https://github.com/LE-Liang/Multimodal-Wireless) is MIT
  licensed; that code license is separate from the dataset's stated CC BY
  license.
- The dataset authors' [beam-prediction paper (arXiv:2603.15093v2)](https://arxiv.org/abs/2603.15093)
  turns this into a concrete decision task: predict the next 10 beam indices
  from a 400 ms history of optimal beam indices, LiDAR, and RGB. Its reported
  setup uses 50 trajectories with a trajectory-disjoint 40/5/5
  train/validation/test split (41,800/6,000/6,000 samples). It reports 80.8%
  average Top-1 beam accuracy and 89.1% normalized gain for a 777M-parameter
  model. The paper's ablation says adding LiDAR to beam history approaches the
  full model, while camera-only addition is weaker but still improves over
  beam-history-only. These are useful task and baseline clues, not independent
  replication or proof that both image and point-cloud inputs are necessary.

## Candidate task and required controls

The strongest candidate is the paper's future beam-selection decision: use
the last 400 ms of beam indices, RGB, and LiDAR to select beams over the next
100 ms. This makes the sensor/video stream consequential to a concrete action
selection task, and gives an existing trajectory-disjoint protocol to
reconstruct. The paper's model is 777M parameters and its published Top-1
result is 80.8%; it does not meet the project's provisional 100–200M
prequantization size range or 95–98% aspiration. A small model should be
evaluated against the exact task and strong non-neural/persistence baselines,
with all results scoped to this simulated beam task.

Before admission, reconstruct and hash the exact 50 trajectories and their
40/5/5 assignments, confirm no trajectory or overlapping window crosses a
split, and freeze the beam-index derivation, tie-breaking rule, input history,
and prediction horizon. Compare beam-history-only, RGB-only, LiDAR-only,
history+RGB, history+LiDAR, and joint inputs with missing, shuffled, stale, and
time-offset controls. The paper's reported modality ablation is motivation,
not sufficient proof of joint-input necessity for our runtime/model.

## Acquisition and next gate

At the last local check, C: had approximately 0.48 GiB free, below the
project's 2-GiB reserve. The OneDrive folder's payload size was not available
from the public page. No download was attempted. Before acquisition, obtain
the official file inventory, per-file sizes, and immutable hashes using the
provider's normal listing interface; choose a scenario subset only after the
full inventory and split plan are known. Do not fetch an unknown-size folder
to the constrained C: drive.

The source remains unadmitted until the derived target, trajectory-grouped
split, payload size/hash inventory, and end-to-end modality controls are
established. No sealed audit or product-quality claim is involved in this
screen.
