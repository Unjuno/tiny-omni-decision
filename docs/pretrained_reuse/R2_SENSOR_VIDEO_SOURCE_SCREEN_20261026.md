# R2 sensor + video source screen — MM-Fit and MMAct

Date: 2026-10-26. This is a metadata and rights screen for two synchronized
video + wearable-sensor candidates. No dataset archives, media, sensor records,
labels, or annotations were downloaded or opened. Neither candidate is admitted
for model training or evaluation by this screen.

## MM-Fit

The official project page describes synchronized RGB-D video, accelerometer,
gyroscope, heart-rate, and magnetometer streams from wearable devices. The
official Zenodo record (DOI `10.5281/zenodo.7672767`, record 7672767) identifies
its payload specifically as **video recordings for 20 workout sessions**, has
21 files totaling 39,109,960,473 bytes, and exposes `CC-BY-4.0` in its record
metadata. The separate sensor data are described on the project page but are
not among the Zenodo video files; no explicit data license for those sensor
files was found in the official material reviewed. The MIT license in the
authors' GitHub repository applies to its starter code and does not establish
rights for the sensor data.

The paper reports 10 human subjects and 616 exercise sets across the collection.
This is a promising task design for sensor + pose/video activity recognition,
but group-independent evaluation has at most 10 subject groups and may have
very wide uncertainty. The paper reports that skeleton-only recognition is
already highly discriminative and that adding sensors gives only marginal
benefit for that task; it therefore does not establish a task where both
modalities are necessary.

**Disposition:** video sub-record `ALLOW` only within the scope of the cited
CC-BY-4.0 Zenodo record; paired video + sensor use remains **REVIEW** until the
sensor-data rights and exact sensor asset/revision are established. Do not
download or use the sensor files yet. Even after rights resolution, use only as
a narrow task study with subject-grouped splits and matched video-only,
sensor-only, and paired controls; it cannot close a broad joint-modality gate.

## MMAct

The pinned official project page source is
[`mmact19/2019@219638d28d3f653e940c32f32b31830e2f655003`](https://github.com/mmact19/2019/tree/219638d28d3f653e940c32f32b31830e2f655003).
The page describes RGB, keypoints, acceleration, gyroscope, orientation, Wi-Fi,
and pressure modalities over 37 classes, but its terms limit use to academic
research and prohibit direct or indirect commercial use and distributing the
dataset or combining it with models/programs for commercial use. It also says
the complete annotations and data are provided on request. Those terms do not
fit the project's product-oriented source gate.

**Disposition:** `RESEARCH_ONLY`; exclude from this product-path training and
quality evaluation. No request for access was sent and no data were fetched.

## Effect on R2

These screens do not close M-2. MM-Fit is potentially usable only after the
sensor-data rights gate and is a small, narrow exercise task; MMAct is
research-only under the inspected official terms. Existing STARSS23 and
Perception Test results retain their previously reported limits and are not
reclassified as proof of a task requiring both modalities. Continue screening
for a rights-cleared, group-splittable, synchronized task with labels that
support a meaningful paired-vs-unimodal comparison. Do not train on either
candidate based on this screen.

## BridgeData V2 follow-up screen

The official [BridgeData V2 project page](https://rail-berkeley.github.io/bridgedata/)
states that its data are CC BY 4.0 and describes 60,096 robot-manipulation
trajectories with language instructions, image observations, 7D robot state,
timestamps, and 7D actions. The official [TFDS catalog](https://www.tensorflow.org/datasets/catalog/bridge)
documents the observation/action fields and a fixed `train`/`test` split of
25,460/3,475 episodes. This makes it a plausible video+proprioception task
source for a separately designed action-choice Decision benchmark; it does not
contain audio and is not evidence that both inputs are necessary.

The exact official code revision inspected was
[`rail-berkeley/bridge_data_v2@bc60a35b701a12021c8c95e9d8601274d3acd928`](https://github.com/rail-berkeley/bridge_data_v2/tree/bc60a35b701a12021c8c95e9d8601274d3acd928).
Its raw conversion script randomly shuffles trajectories independently inside
each dated data-collection folder, then assigns 90%/10% to train/validation;
the inspected code does not fix the random seed. Therefore that raw split is
not an immutable reproducible split, and it does not establish collection,
environment, or scene-group isolation. A new deterministic split would need
to group by collection/session and likely environment before any model fitting.

Storage is the limiting factor for an actual data pilot: the official raw
archives are 411 GB and 30 GB, while the TFDS catalog reports 387.49 GiB. The
TFDS web directory exposes many roughly 90–140 MB shards, but the current local
C: drive has only 39.4 GB free. No BridgeData shard or archive was fetched,
and no data contents were inspected. A curated shard subset cannot be assumed
to have source-group-disjoint train/validation membership until episode
metadata and source paths are audited.

**Disposition:** `REVIEW / metadata-only candidate`. It is the strongest
currently identified rights-cleared synchronized video+state source, but do
not acquire shards or train yet. First define an action-choice formulation
whose label genuinely depends on both the frame sequence and proprioceptive
state, establish immutable collection-group splits from official episode
metadata, measure the download footprint for only those groups, and retain
matched video-only/state-only plus shuffle/stale controls. No experiment or
quality claim is made here; R2 M-2 remains open.

## Sources inspected

- [MM-Fit official project page](https://mmfit.github.io/)
- [MM-Fit official Zenodo video record, DOI 10.5281/zenodo.7672767](https://zenodo.org/records/7672767)
- [MM-Fit paper](https://doi.org/10.1145/3432701)
- [MM-Fit authors' starter-code repository](https://github.com/KDMStromback/mm-fit)
- [MMAct official project terms, pinned source commit](https://github.com/mmact19/2019/blob/219638d28d3f653e940c32f32b31830e2f655003/index.html)
- [BridgeData V2 official project and license](https://rail-berkeley.github.io/bridgedata/)
- [BridgeData V2 official TFDS schema and split counts](https://www.tensorflow.org/datasets/catalog/bridge)
- [BridgeData V2 official code, pinned revision](https://github.com/rail-berkeley/bridge_data_v2/tree/bc60a35b701a12021c8c95e9d8601274d3acd928)
