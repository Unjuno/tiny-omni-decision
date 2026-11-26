# EgoInertia-MI source screen — 2026-10-29

## Decision

Keep `REVIEW_NOT_ADMITTED`. This is a strong task-shape candidate for a narrow
video-plus-sensor Decision study, but the dataset is access-restricted and no
explicit data license is identified on the official project page. No access
request was sent and no payload was obtained or inspected.

## Official project-page evidence

The [EgoInertia-MI project page](https://fatemah-alh.github.io/EgoInertia-MI-Page/)
describes 19 daily and clinical-style activities, chest-mounted GoPro HERO12
video at 30 FPS, and wrist, leg, and chest IMUs sampled at 25.5 Hz. Video and
IMU are organized by participant, session, and recording with aligned IDs.
The stated tasks are 19-class action recognition and 3-class severity
estimation using natural, mild, and severe *simulated* impairment conditions.

The project page specifies five-fold subject-disjoint evaluation with 12 train,
2 validation, and 3 test subjects per fold. It also says the dataset is
available through restricted Zenodo access and asks users to provide their
name, affiliation, and intended research use. The page requires adherence to a
dataset license but does not identify a license name or provide terms in the
inspected content. The project describes the data as a research benchmark, not
for direct clinical diagnosis.

## Fit and unresolved gates

This candidate provides an unusually useful design for testing video and
wearable-sensor inputs under participant-disjoint splits. The supplied
classification tasks can be converted into fixed-option Decision questions,
but they do not by themselves establish that both modalities are necessary.
A future admitted study must preregister video-only, IMU-only, and joint
controls, modality perturbations, task metrics, and a split derived from the
official subject IDs. Report action and simulated-severity performance
separately, without implying clinical diagnostic quality.

Before any use, resolve all of the following through authorized public terms
or a user-approved access path:

1. Dataset payload license and attribution requirements.
2. Access conditions and whether model-training/derived artifact use is allowed.
3. Participant consent, privacy constraints, retention, and redistribution rules.
4. Exact release version, annotation/split files, and checksums.
5. Payload sizes and storage feasibility before requesting or downloading data.

Because the official flow requires a request containing personal affiliation
and intended-use details, this screen did not submit one. Do not treat the
project code or paper license as a dataset license.

## Reproducibility

Metadata-only screen from the official project page on 2026-10-29. No files,
media, model weights, or sealed audit were accessed; no training was started.
See the [machine-readable screen](../../manifests/pretrained_reuse/egoinertia-mi-source-screen-20261029.json).
