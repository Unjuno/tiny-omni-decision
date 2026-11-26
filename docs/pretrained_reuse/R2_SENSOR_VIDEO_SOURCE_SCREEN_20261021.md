# R2 synchronized sensor/video source screen — 2026-10-21

## Decision

No reviewed source is admitted to model fitting or quality selection. CUHK-X is
the strongest engineering candidate for a later, explicitly exploratory
cross-subject pilot if its Data Use Agreement is acceptable and a specific
both-input task is preregistered. It has only 30 participants, so it does not
meet the roadmap's provisional 200-independent-group release-quality floor.
The other two candidates have unresolved dataset rights and fewer independent
participants. No media, labels, or archives were downloaded or opened; no
access agreement was accepted; no email/access request was sent; no model ran.

## CUHK-X

Official source: [repository and dataset README](https://github.com/openaiotlab/CUHK-X).

- The official README describes 64,267 samples from 30 participants, with
  synchronized RGB, infrared, thermal, depth, mmWave radar, skeleton, and IMU
  streams. It includes cross-subject HAR and sequential action reasoning.
- The official release says dataset files are distributed separately under a
  Data Use Agreement, for non-commercial research, and are not redistributable.
  The repository's MIT license applies to code; derived annotations/splits have
  a separate CC BY 4.0 statement. These terms do not establish permission for
  product training or redistribution of source media.
- The published small-model challenge split trains on users 1–9 and 16–24 and
  tests on users 10–11 and 25–26. That is a useful cross-subject precedent,
  but only four test subjects and no proof that an individual label requires
  two streams. Challenge test labels/private evaluation remain out of scope.
- A new development-only task would need train/validation subjects disjoint at
  source level, a comparison against each matched unimodal baseline, and
  missing/shuffled/stale controls. Multimodal availability or synchronized
  timestamps alone is not evidence of complementarity.

Disposition: **conditional research candidate only**. Do not accept the DUA,
download, or use the benchmark until access and intended-use compatibility are
resolved and a both-required task is fixed. Even then, treat 30 participants as
low-power exploratory evidence, not as satisfying the provisional group floor.

## C-MHAD

Official sources: [UT Dallas dataset page](https://personal.utdallas.edu/~kehtar/C-MHAD.html),
[dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7287800/).

- The publisher describes synchronized 15-fps video and 50-Hz wearable
  acceleration/gyroscope streams, 240 two-minute recordings from 12 people,
  with manually time-marked gesture and transition events in continuous
  streams. Its page says the data are available to the research community and
  links to a Box download.
- The paper is published under CC BY, but the inspected official dataset page
  does not state a separate license for the recordings. The paper's license is
  not treated as a data license.
- The paper reports improved F1 from simultaneous/fused sensing, which is
  motivation for a candidate, not independent evidence that every target
  decision requires both modalities. The 12-person cohort is far below the
  provisional group floor.

Disposition: **not admitted** pending an authoritative data-use grant and a
predeclared both-required task; too few independent participants for a quality
claim. Do not download the recordings while data rights remain unclear.

## XRFv2 Plus

Official sources: [project dataset page](https://airslab2020.github.io/labpages/datasets/xrfv2-plus/index.html),
[Zenodo record](https://zenodo.org/records/20564312).

- The project page reports 853 aligned sequences, 16 volunteers, 34 action
  classes, 8,157 temporal segments, and 108,929 QA pairs. It includes five
  position IMU, AirPods IMU, Wi-Fi CSI, one-per-second RGB video features,
  depth/infrared video, and body representations. Raw RGB is excluded from the
  public package for privacy.
- The Zenodo Rights/License field is blank in the inspected record. The project
  page does not establish a component-level license for sensor data, annotations,
  RGB-derived features, or depth/infrared media. The original repository's code
  license is not assumed to cover these data.
- Sixteen volunteers are insufficient for the provisional independent-group
  quality floor. Repeated QA rows must not be counted as independent examples
  when they share a sequence.

Disposition: **not admitted**. Rights, component provenance, encoder-training
overlap, and a both-required task remain unresolved; the subject count is low.

## Next gate

For any of these sources, first establish exact data-use rights and immutable
TRAIN/validation revisions. Then define one temporal decision whose correct
answer cannot be obtained from either input alone, freeze subject/scene groups,
and specify unimodal, joint, missing-input, shuffle, stale-input, and
time-offset controls. Only proceed to a bounded frozen-feature pilot after
these gates pass and local storage/runtime preflight succeeds. Keep any
challenge/test or sealed data unopened. Sensor remains optional; this screen
does not delay the four required modalities or close R2 M-2.
