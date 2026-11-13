# C-MHAD synchronized video/IMU source screen (2026-10-28)

Status: **REVIEW — no data downloaded and no model fit**.

## Why it is relevant

The official C-MHAD page describes continuous 2-minute video and inertial
streams captured at the same time, with manually marked action intervals. The
two task families are (1) five smart-TV hand gestures and (2) seven transitions
among standing, sitting, lying, and falling. Each family has ten streams from
each of 12 participants (240 streams total). The video is 640x480 at 15 fps;
the six-axis Shimmer3 inertial data are 50 Hz. The gestures are performed among
unconstrained non-target actions, so a real-time action/transition Decision
task with supplied alternatives is plausible. A useful comparison would need
video-only, IMU-only, and joint readouts plus timestamp shifts, stream swaps,
and no-event/metadata controls.

## Admission blockers

- The official dataset page says the dataset is made available to the research
  community and asks users to cite the paper, but it does not state a license
  for the downloadable videos, inertial files, or annotations. The paper is
  published under CC BY, which is not evidence that the separate data payload
  has the same license. The download button links to a Box share; no payload was
  opened or fetched.
- The videos contain identifiable participants. The official page does not
  specify participant release/redistribution terms or a separate data-use
  agreement. Privacy and downstream model-use scope therefore remain unknown.
- The independent group count is only 12 people. A strict participant-held-out
  split would leave very small validation and audit cohorts. This can support a
  narrow exploratory diagnostic only if rights are clarified; it cannot by
  itself support broad action understanding or the product quality target.
- The source has no public checksum/immutable data revision on the dataset
  landing page. These must be pinned and verified before any acquisition.

## Decision and next gate

Do not fetch or train on C-MHAD yet. Before admission, the rights holder must
publish or provide terms covering the media, annotations, and model-training
use, and the source must be pinned with a reproducible checksum. No contact was
sent. If admitted later, first audit participant IDs, stream alignment, action
intervals, no-event intervals, and duplicate scenes, then preregister the
subject-group split and option-only baseline before extracting features.

## Primary sources

- [Official C-MHAD dataset page](https://personal.utdallas.edu/~kehtar/C-MHAD.html)
- [C-MHAD paper](https://doi.org/10.3390/s20102905) (the paper's license does
  not establish a license for the separate dataset payload)
