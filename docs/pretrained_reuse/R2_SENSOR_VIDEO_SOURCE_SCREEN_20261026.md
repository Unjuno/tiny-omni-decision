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

## Sources inspected

- [MM-Fit official project page](https://mmfit.github.io/)
- [MM-Fit official Zenodo video record, DOI 10.5281/zenodo.7672767](https://zenodo.org/records/7672767)
- [MM-Fit paper](https://doi.org/10.1145/3432701)
- [MM-Fit authors' starter-code repository](https://github.com/KDMStromback/mm-fit)
- [MMAct official project terms, pinned source commit](https://github.com/mmact19/2019/blob/219638d28d3f653e940c32f32b31830e2f655003/index.html)
