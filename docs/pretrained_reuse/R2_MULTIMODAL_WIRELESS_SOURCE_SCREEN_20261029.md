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

## Candidate task and required controls

A plausible task to test is a future-link decision: choose a discrete
transmission action (for example, a beam or channel-quality class) at a future
frame from current visual context plus current/previous sensor or CSI context.
This is a proposed derived target, not a label documented by the dataset.
Before admission, define the target deterministically from the released
channel fields, pin the horizon and tie-breaking rule, and verify frame and
agent alignment. Compare camera-only, sensor/CSI-only, and joint input using
scenario-grouped splits, plus missing, shuffled, stale, and time-offset input
controls. The task only establishes usefulness for this narrow simulated V2X
decision; it cannot establish general multimodal decision quality.

## Acquisition and next gate

At the last local check, C: had approximately 0.48 GiB free, below the
project's 2-GiB reserve. The OneDrive folder's payload size was not available
from the public page. No download was attempted. Before acquisition, obtain
the official file inventory, per-file sizes, and immutable hashes using the
provider's normal listing interface; choose a scenario subset only after the
full inventory and split plan are known. Do not fetch an unknown-size folder
to the constrained C: drive.

The source remains unadmitted until the derived target, grouped split, payload
size/hash inventory, and end-to-end both-input controls are established. No
sealed audit or product-quality claim is involved in this screen.
