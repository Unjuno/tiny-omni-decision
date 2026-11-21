# MotorSense video/EMG/IMU source screen (2026-11-21)

## Decision

`REVIEW_NOT_ADMITTED` for training or quality selection. The task fit is
interesting, but the official dataset Hub marks the data license as `other`
without publishing specific data-use terms. The separate GitHub repository's
MIT license is for the code; it does not grant rights to the media, signals, or
annotations. No payload, metadata file, license acceptance, or account-gated
resource was accessed.

## Task fit and scale

The [publisher project page](https://www.cs.umd.edu/~edessale/motorsense/) and
[official dataset repository](https://huggingface.co/datasets/edessa/motorsense)
describe 370 recordings with synchronized egocentric video, bilateral surface
EMG and IMU, and action/contact labels. The project reports 5,209 action
segments on 85 recordings and 4,700 per-hand contact intervals on 57
recordings. This could support a temporal video-plus-sensor decision, but the
labels alone do not show that both inputs are necessary. A suitable task would
need recording/person-grouped splits and video-only, sensor-only, paired,
missing-input, temporal-shift, and cross-recording controls.

## Admission blockers

- The [Hub dataset card](https://huggingface.co/datasets/edessa/motorsense)
  displays license `other` and contains no specific license terms in the card.
  The [code repository](https://github.com/edess442/MotorSense) advertises an
  MIT code license; it does not establish a license for the dataset content.
- The publisher reports approximately 0.9 TB total data (about 840 GB of
  frames). It provides a metadata-only download option of a few MB, but no
  metadata or media was downloaded during this screen.
- The project explicitly notes that faces are not blurred. Permission and
  participant-consent scope for downstream model training and product use are
  not established by the displayed Hub license tag.
- The release provides train/validation lists for action intervals, but action
  data cover only 85/370 recordings; contact data cover 57 recordings. Actual
  recording/person group counts and content leakage would need an authorized
  metadata-only audit before a task can be frozen.

## Next gate

Do not download the data or use its annotations until an authoritative
dataset-content license and consent scope are published or provided. Even if
rights are clarified, first check whether a small, person-disjoint group split
is feasible from approved metadata; do not fetch the full archive. The current
four-modality objective remains open. HubEgo is still the preferred video-plus-
sensor candidate in the roadmap, pending the user's account-level approval.
