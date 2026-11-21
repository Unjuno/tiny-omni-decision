# EVIORA-HAR video + IMU source screen — 2026-11-21

## Decision

`NOT_ADMITTED_PRODUCT`: the Zenodo release explicitly declares CC BY-NC 4.0,
which is incompatible with this product-directed model-development objective.
The released source is also about 131.3 GB. Do not download or use the payload
for this workstream. This was a metadata-only screen; no data files, labels,
videos, or consent-form archive were opened.

## Publisher evidence

- Zenodo record `22239523`, DOI `10.21227/2zvc-nb24`, published 2026-08-09.
  Its public API declares `cc-by-nc-4.0` and lists 131,345,843,274 bytes across
  split archives, README, machine metadata, scripts, and a consent-form ZIP.
- The data descriptor describes 104 participants, 71 older adults in ADL
  recordings and 33 young adults in safety-controlled fall recordings, with
  30-channel wearable IMU and up to three camera views. It reports 3,489
  complete four-modality trimmed segments, but the two acquisition groups do
  not have frame-level synchronization across all modalities.
- The descriptor article itself is CC BY-NC; this matches, but is independent
  of, the Zenodo dataset license. The dataset's own API license is decisive
  for this screen.

## Interpretation

This is a clinically relevant sensor/video task source, but it cannot be
admitted for a product-directed training/evaluation path under the stated
license. The dataset also mixes older-adult ADL recordings with young-adult
simulated falls; it cannot be represented as a single population-general
fall benchmark. These details do not change the license decision. No clinical
deployment claim is made.

## Evidence and provenance

- [Zenodo record](https://zenodo.org/records/22239523)
- [Read-only Zenodo API metadata](https://zenodo.org/api/records/22239523)
- [Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC13654039/)
- Source commit reviewed: `1dfea4e61c86aa76b39bccd6ec1fd97e7612afd1`.
- Payload download: none. Consent-form archive and labels: not opened. Model
  work, GPU use, and cloud cost: none.
