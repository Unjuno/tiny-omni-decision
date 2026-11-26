# WEAR video and IMU source screen — 2026-10-29

## Decision

`NOT_ADMITTED` for product-directed training. The modality/task fit is good,
but the official dataset license is CC BY-NC-SA 4.0, which excludes commercial
use and imposes ShareAlike obligations. The raw payload is also about 130 GB,
far above available storage. No dataset data or media was downloaded.

## Official evidence

The [official WEAR dataset page](https://mariusbock.github.io/wear/) describes
synchronized egocentric video and inertial data for 18 outdoor workout
activities by 22 participants at 11 locations. It lists per-subject annotation
JSON files (>1 MB), processed feature files (about 15 GB), and raw per-subject
video/IMU files (about 130 GB). The same page states CC BY-NC-SA 4.0 terms,
including the non-commercial restriction and ShareAlike condition.

The dataset could be useful for an explicitly non-commercial research benchmark
of video/IMU activity recognition. That does not establish suitability for the
intended product, and class labels alone do not prove that both modalities are
necessary. The public page inspected here does not establish an independent
participant-held-out protocol; a group split and matched unimodal controls
would still need verification before a research-only study.

## Reproducibility and limits

Metadata-only review of the official dataset page on 2026-10-29. No archive,
feature, or media payload was fetched. No training or sealed-audit access.
See the [machine-readable screen](../../manifests/pretrained_reuse/wear-video-imu-source-screen-20261029.json).
