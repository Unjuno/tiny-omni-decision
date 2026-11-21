# MUVS video + device-sensor source screen — 2026-11-21

## Decision

`REVIEW_NOT_ADMITTED`: MUVS is a promising small-scope camera-selection
Decision task, but the authoritative records conflict on whether the *video
payload* is covered by an open license. Do not download annotations or videos,
extract features, fit a model, or use its labels until the provider clarifies
the grant. This is a source screen only; no task labels or media were opened.

## Pinned publisher facts

- Zenodo record `20708683`, v1.0, published 2026-06-15, DOI
  `10.5281/zenodo.20708683`. Its public API metadata identifies the deposited
  record as CC BY 4.0. The 9.74 MB deposit contains `data.zip`, README,
  downloader and requirements file.
- The accompanying README explicitly says `License: [TO BE ADDED]` and that
  the period videos are hosted separately in public GCS, not in the Zenodo
  deposit. The official project website reserves copyright and does not state
  a video license. The Zenodo license therefore cannot safely be assumed to
  cover those separately hosted videos. Face blur is described, but does not
  itself establish copyright or participant-use permission.
- The README describes 17 live sports events, 5,039 three-second fragments,
  258 period videos, 2–3 simultaneous cameras, and 18 device-sensor channels.
  It reports train/test splits at event level (11/6), with annotations for
  camera selection. Full raw video is about 164.9 GB; data can be downloaded
  by selected event. Do not inspect the publisher test labels.

## Potential task fit and unresolved checks

The proposed task is: given synchronized camera clips and their device
telemetry, select the best camera for a short event fragment from stable
camera-ID options, returning calibrated probabilities. This is a natural
Decision interface. It may test video + sensor complementarity, but the
annotation's actual criteria and agreement are not described in the metadata
screen, and the task may be solved from video alone. A future admission must
freeze an event-grouped split using only eligible publisher-train events,
verify camera/period/fragment identities and source hashes, exclude the
publisher test partition, and compare video-only, sensor-only, and paired
systems with video/sensor shuffling, missing-input and temporal-offset
controls. It must also document annotator agreement and ensure no camera/event
identity shortcut predicts the selected option.

The event count is small for independent generalization. A 164.9 GB full-video
download is unnecessary for a first feasibility pilot, but selected-period
sizes and hashes must be inventoried before any payload acquisition. Current
C: free space is sufficient; this does not waive the rights gate.

## Evidence and provenance

- Zenodo record and file listing: <https://zenodo.org/records/20708683>
- Read-only Zenodo API metadata: <https://zenodo.org/api/records/20708683>
- Publisher README preview: <https://zenodo.org/records/20708683/preview/README.md?include_deleted=0>
- Official dataset site: <https://swperfi-project.github.io/Pages-dev/MuvsDataset-page/>
- Reviewed 2026-11-21 from source commit `4ba5313eda00fc2b4b7bb85b19fa9f7e4c0d94e2`.
- No payload was downloaded; no annotation files, test labels, videos, or model
  outputs were opened. No training, GPU work, or cloud cost.

**Re-entry condition:** obtain a written provider clarification that the
separately hosted video files and annotations are covered by a license
permitting the intended product-directed local model development, and record
the grant and attribution requirements. Otherwise reject the candidate.
