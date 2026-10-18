# C-MHAD synchronized video and inertial source screen — 2026-10-19

## Decision

C-MHAD is a strong **task-shape** candidate for temporal video + sensor
decisions: its official page describes continuous action streams with manually
annotated event intervals, synchronized camera video, and six-axis wearable
inertial measurements. It is **not admitted** to training or evaluation because
the official dataset page does not state the corpus license or explicitly
authorize derivative-model training, commercial use, or media redistribution.
It also does not publish a model-selection split. No data were downloaded,
opened, decoded, or evaluated.

## Source facts

- Official source: [C-MHAD project page](https://personal.utdallas.edu/~kehtar/C-MHAD.html).
- The page describes 240 two-minute streams from 12 participants: 120 smart-TV
  gesture streams and 120 transition-movement streams.
- Video is 640×480 at 15 fps; Shimmer3 acceleration and angular velocity are
  sampled at 50 Hz. The source describes timestamp synchronization, while
  noting that 30–40 early IMU samples can be missing because of Bluetooth
  latency and may need leading zero padding.
- Five gesture classes and seven transition classes have manually annotated
  start/end times. This could support event detection and candidate scoring on
  temporal windows, with video-only, sensor-only, and joint controls.
- The page links a Box archive and Python/MATLAB readers, but does not give a
  dataset-level license or specify commercial/model-training rights. The
  article's open-access publication license does not establish reuse rights for
  the separately hosted corpus.
- No official train/validation/test split is specified. A defensible split
  would group all recordings by participant; 12 total participants limit the
  number of independent validation groups and any tight uncertainty claim.

## Admission conditions

Before reconsidering this source, require explicit dataset-level rights from
the authors/source, a pinned archive and component hashes, and a predeclared
participant-grouped split with train-only normalization. Then test that
targets remain valid under video-only, sensor-only, missing, shuffled, stale,
and timing-offset controls. The set of windows must be generated without
crossing participant splits, and interval labels near the 30–40-sample IMU
startup gap need a declared policy.

Until those conditions are met, keep the source out of all catalogs. This screen does not change the existing Lombard,
STARSS23, CLEVR-4, or Teacher artifacts and does not justify a new GPU run.

## Provenance

The source facts above were checked against the official project page on
2026-10-19. The associated paper is [Wei, Chopada, and Kehtarnavaz, *C-MHAD:
Continuous Multimodal Human Action Dataset of Simultaneous Video and Inertial
Sensing*](https://doi.org/10.3390/s20102905). No archive bytes or labels were
read. Machine-readable disposition:
`manifests/pretrained_reuse/c-mhad-source-screen-20261019.json`.
