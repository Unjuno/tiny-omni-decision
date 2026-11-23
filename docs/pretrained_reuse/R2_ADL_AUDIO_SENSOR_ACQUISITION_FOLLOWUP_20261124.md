# ADL audio/sensor source acquisition and identity audit — 2026-11-24

## Decision

The source is now locally available and the official Zenodo API metadata confirms
`CC-BY-4.0`. Keep it **REVIEW_NOT_ADMITTED_FOR_QUALITY_TRAINING**. It can support
a narrow audio/sensor activity-recognition diagnostic, but it has no video,
only 14 human participants, and highly uneven class support. In particular, the
231 `no-activity` records carry participant ID `999` in every device file, so
they cannot be assigned by the same participant-group split as the human
activity records. No valid split has been frozen and no model or signal payload
has been evaluated.

## Acquisition and integrity

- Official source: [Zenodo record 7937591](https://zenodo.org/records/7937591),
  DOI `10.5281/zenodo.7937591`, version `1.0`; the current record API names
  license `cc-by-4.0`.
- All 25 activity ZIPs (4,572,316,824 bytes compressed) were downloaded to
  `C:\CodexArtifacts\adl-audio-sensor-source-screen-20261124-attempt-01`.
- The downloader receipt reports all 25 files as `downloaded_verified`. A
  second local pass checked every archive's exact API size and MD5: 25/25 pass,
  zero mismatches, zero `.part` files. ZIP central-directory inventory found
  125 HDF5 members and 12,288,738,794 uncompressed bytes.
- Dataset README SHA-256:
  `d5fe3c9097226f00656a65b86d8374a4bf62c42f8c80b156cbe2a69ab33f83ac`.
- Zenodo record JSON SHA-256:
  `7a84948e27d6193084973fde5743b591dc0e1b2eb8e633729e929d30684c356a`.
- Download receipt SHA-256:
  `e485cc5be38326529c038a6fcfade27f7fd18eaa2e9e8952cd9af15c1c3d621d`.
- The extracted HDF5 files and compact metadata inventory remain outside Git.
  Participant, trial, timestamp, and HDF5 structure metadata were read. In one
  `no-activity` file, ragged audio arrays were loaded only to count samples and
  infer their approximate 60-second duration; waveform values were not
  inspected or used. Vibration, IR, light, and environmental signal values
  were not read or processed.

## Identity and support findings

All five device files for each activity contain identical participant/trial
pair sets, which supports joining synchronized rows across devices without
cross-device split drift. The inventory contains 8,615 activity/trial groups
per device and 14 distinct human participant IDs. The 15th ID (`999`) occurs
only in `00_no-activity` and is a nonparticipant sentinel in the archive.

| Activity | Trial groups | Human participants |
|---|---:|---:|
| no-activity | 231 | 0; sentinel 999 |
| walk | 464 | 11 |
| walk-to-room | 373 | 9 |
| open-door | 448 | 11 |
| close-door | 454 | 11 |
| open-window | 465 | 11 |
| close-window | 465 | 11 |
| sit-down | 466 | 11 |
| stand-up | 460 | 11 |
| light-on | 384 | 9 |
| light-off | 369 | 9 |
| typing | 14 | 9 |
| make-coffee | 72 | 9 |
| place-plate | 370 | 9 |
| sweep | 9 | 7 |
| vacuum-cleaning | 14 | 8 |
| start-toaster | 373 | 8 |
| stop-toaster | 368 | 8 |
| open-fridge | 355 | 8 |
| close-fridge | 359 | 8 |
| open-dishwasher | 453 | 11 |
| close-dishwasher | 453 | 11 |
| place-on-stove | 380 | 9 |
| take-from-stove | 368 | 9 |
| wipe-stove | 448 | 11 |

The tiny `typing` and `sweep` classes (14 and 9 trials) make a reliable 25-way
quality estimate implausible. A random row split would leak participant and
trial identity. A participant-disjoint split alone would place every
no-activity row into one partition because it has only the sentinel ID. A
separate time-block grouping for those 231 rows may be possible, but has not
been validated; it must also avoid adjacent-window leakage and preserve a
training example for every reported class.

### Split feasibility check

For the 24 human-activity labels, an exhaustive label-support feasibility
count over all `8 train / 3 validation / 3 sealed-audit` assignments of the 14
participants found 60,060 possible assignments. Of these, 42,566 have at least
one trial for every class in every partition, 14,430 have at least two, and
830 have at least three. This establishes only that a participant-disjoint
partition is combinatorially possible; it does not make the 14 participant
clusters statistically sufficient, and no assignment was selected.

The `no-activity` rows are each approximately 60 seconds long. Their timestamps
form 13 contiguous runs when separated at gaps over 300 seconds (run sizes
10, 10, 113, then nine runs of 10 and a final run of 8). The largest run
dominates this class, and the 13 runs still come from one recording
environment. The 300-second threshold was used only to describe the observed
metadata; it is not a frozen split rule. A defensible time-block rule and
independent-group count would need to be fixed before any model evaluation.

Given the small number of independent human participants, very sparse classes,
and only 13 candidate background time blocks, this source cannot currently
support the project's broad high-quality or calibrated multimodal claim. It
may support a clearly labeled exploratory sensor/audio fusion study only if a
predeclared grouped protocol and matched unimodal controls are justified.

## Fit to the product objective

This is optional sensor evidence, not a replacement for the required text,
image, audio, and video gates. Its single kitchen/dining-room environment and
small number of participants sharply limit generalization. The source's
activity labels also do not prove that audio and another sensor are jointly
needed. Any future diagnostic must freeze participant/trial groups first and
compare audio-only, vibration-only, other-sensor-only, and joint-input
controls, including missing and mismatched streams. Calibration and
participant-clustered uncertainty matter as much as accuracy.

This source audit alone does not justify a model run. Prioritize the project's
required text/image/audio/video Path A evidence; retain this dataset as an
optional exploratory audio/sensor source rather than spending model-training
budget on it by default.

## Reproducibility

The machine-readable acquisition record is
[`adl-audio-sensor-acquisition-20261124.json`](../../manifests/pretrained_reuse/adl-audio-sensor-acquisition-20261124.json).
The complete local participant/trial inventory is at
`C:\CodexArtifacts\adl-audio-sensor-source-screen-20261124-attempt-01\participant-trial-inventory.json`
(SHA-256 `ddd9b982fa0ad74d98638812a042c1daa0e2402a02c84023ef371eba8964ebcf`).
The data bundle was not added to Git. No model was trained or evaluated, no
validation/audit partition was inspected, and no GPU was used.
