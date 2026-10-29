# R2-M video and sensor source recheck — MM-Fit and Multimodal-Wireless (2026-10-30)

## Disposition

Neither source is admitted for a joint video/sensor fit yet. MM-Fit has a
clear license for its Zenodo video payload, but the companion inertial archive
does not have explicit terms on the official project page. Multimodal-Wireless
states CC BY 4.0 for its dataset and has synchronized simulated sensor streams,
but its downloadable data is shared from a mutable OneDrive folder and the
published landing page does not pin a release archive/hash or specify a
decision-label protocol. No dataset payload was downloaded and no model was
trained.

## MM-Fit

The authors' project page describes synchronized multi-view RGB-D video and
wearable inertial streams, ten exercise categories, and over 800 minutes of
data from five sensing devices. Its download link for the companion dataset
points to an S3 `mm-fit.zip` archive, but the page does not state a license or
data-use terms for that inertial/pose payload. The separate Zenodo v2 video
record (DOI `10.5281/zenodo.7672767`) has an explicit `CC-BY-4.0` license in
its API metadata and lists 39.1 GB of RGB video files. Individual videos range
from approximately 396 MB to 3.1 GB in the record's displayed file sizes. This
video license does not establish rights for the separate inertial archive.

This could support a narrow exercise-recognition comparison if the companion
sensor rights are clarified and participant identities can be grouped across
splits. The structured workout sessions also require leakage controls against
workout order and session position. The current C: volume has
about 2.21 GiB free, insufficient to safely stage even the full RGB video set,
and there is no other local filesystem volume available in this environment.

## Multimodal-Wireless

The publisher describes a CARLA/Sionna-generated dataset with 161,400 frames
across four virtual towns, 16 communication scenarios, and three weather
conditions. It reports synchronized RGB/depth, IMU, LiDAR, radar, and wireless
channel measurements, and states that the dataset is CC BY 4.0. This is a
potentially useful controlled-simulation task for testing whether visual and
sensor features jointly predict a future driving state.

The release is not yet reproducibly pinned: the publisher's GitHub README
directs users to a shared OneDrive folder for both town assets and scenario
data, while the dataset page does not provide an immutable data revision,
per-file checksums, subject/scene split manifest, or task-label schema. The
linked repository's `main` is also mutable. The code repository describes
scenario configuration and data generation, but no canonical decision target
or leak-resistant train/validation split has been frozen for this project.
Before data acquisition, require a pinned source snapshot with hashes and
define a train-only target derived from simulator state, grouped by town and
scenario seed, with timestamp-shift and modality-ablation controls.

## Source and local-state evidence

- [MM-Fit project page](https://mmfit.github.io/): synchrony, modalities,
  exercise types, and official download links.
- [MM-Fit Zenodo v2 record](https://zenodo.org/records/7672767): video file
  list and record metadata. Read-only API metadata reports `cc-by-4.0` and
  per-file sizes/checksums.
- [Multimodal-Wireless publisher dataset page](https://le-liang.github.io/mmw/):
  dataset description and CC BY 4.0 statement.
- [Multimodal-Wireless publisher repository](https://github.com/le-liang/Multimodal-Wireless):
  mutable data links and generation/replay code. The source inspection did not
  clone or modify this repository.
- Local C: free space at the screen: 2.21 GiB. No other filesystem volume was
  mounted. No dataset media was fetched.

## Next gate

Ask the MM-Fit data maintainers to publish or provide terms for the companion
inertial/pose archive, or obtain a pinned Multimodal-Wireless snapshot and
freeze its task/split protocol. Until then, retain the existing R2-M gate and
do not claim a clean multimodal quality result from either source.
