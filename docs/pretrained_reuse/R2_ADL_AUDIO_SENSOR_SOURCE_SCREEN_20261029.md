# ADL ambient audio and sensor source screen — 2026-10-29

## Decision

Keep this source at `REVIEW_NOT_ADMITTED`. It may be useful for a narrow
audio/environmental-sensor activity-recognition diagnostic, but it has no video
and cannot advance the required video-capable product gate. No archive or
sample was downloaded, no media was inspected, and no model was trained.

## Source evidence

- [Zenodo record 7937591](https://zenodo.org/records/7937591) describes 25
  activities and five multisensor devices, with ambient audio and vibration as
  the primary streams plus infrared array, light/color, and environmental
  readings. Its record lists a total archive size of 4.6 GB; the smallest
  activity archive is 69.2 MB.
- The [dataset paper](https://www.mdpi.com/2306-5729/9/12/144) reports 8,615
  activity instances from 14 participants, individually performing 25
  activities in supervised sessions. It says recordings are synchronized and
  identifies the dataset license as CC BY 4.0.
- The Zenodo page's rendered Rights/License field is blank. The paper's license
  statement is positive evidence but does not resolve the missing record-level
  payload terms. The Zenodo API metadata request timed out during this screen.
  Confirm the exact payload license and attribution requirements before any
  use.
- The public description says participants repeated each activity, but the
  inspected record does not provide a declared train/validation/test split or
  establish that stable participant IDs are available in each archive. Do not
  split random instances: first establish participant identity and enforce
  participant-disjoint partitions.

## Fit to the roadmap

The data could support a task-specific audio/sensor comparison if participant
identities, a valid independent split, and exact payload rights are confirmed.
The activity labels alone do not prove that both audio and sensor streams are
needed: many activities can have strong cues in one stream. Any later
experiment must freeze a task and compare audio-only, sensor-only, and joint
inputs, with missing/shuffled-input controls and participant-grouped
uncertainty.

This is not an Omni source: it has no video, and its supervised single-room
setting and 14 participants are a narrow population. The archive is also not
fetched because the current C: free space is below the project's 2-GiB reserve;
the total archive exceeds remaining free space. Recheck storage and rights
before acquisition. Do not treat the paper's CC BY license alone as permission
for a different version or any unlisted payload.

## Reproducibility and limits

This was a metadata-only screen using the Zenodo record and dataset paper on
2026-10-29. No immutable archive checksum, participant-level split, or
payload-level license metadata was captured. The machine-readable summary is
[`manifests/pretrained_reuse/adl-audio-sensor-screen-20261029.json`](../../manifests/pretrained_reuse/adl-audio-sensor-screen-20261029.json).
