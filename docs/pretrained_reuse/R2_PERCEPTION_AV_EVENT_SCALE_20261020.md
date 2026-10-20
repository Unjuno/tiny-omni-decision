# Perception AV event-source coverage scale experiment

## Status

**Completed 2026-10-21; execution passed, quality conclusion remains uncertain.** This is a narrow development experiment, not a general
multimodal benchmark and not evidence that the four-modality product target has
been reached. The question is whether more unique video groups improve the
existing conditional audio-event to visible-object-source task.

The 200-video fixed-checkpoint holdout already evaluated on 2026-10-20 remains
excluded from training, selection, calibration and the new development view.
Its plan, evaluation receipt, and two prediction files are hash-pinned by the
new planner. The official TRAIN annotations had previously been parsed during
feasibility work, so this cohort is development evidence rather than blind
data. Public annotations do not expose participant identity; participant-level
disjointness remains unknown.

## Frozen cohort

Source: pinned Perception Test TRAIN archive generation
`1685446268139925`, with the existing annotation and cut-map SHA-256 pins.
Earlier media IDs and the already-scored 200 holdout videos are excluded.
Selection and split assignment use seed 17 and deterministic SHA-256 ordering
of video IDs; questions/events from one video remain in one split.

| Split | Videos | Candidate events | Planned uncompressed media |
|---|---:|---:|---:|
| Fit | 233 | 550 | 3,394,165,537 bytes |
| Selection | 58 | 121 | 839,220,475 bytes |
| Calibration | 39 | 91 | 610,747,864 bytes |
| Development validation | 58 | 146 | 848,214,961 bytes |
| **Total** | **388** | **908** | **5,692,348,837 bytes** |

The reserved free-space requirement is 6,830,818,604 bytes (1.2 times planned
media). The C drive had 44,151,877,632 free bytes at the last preflight.

Frozen plan:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-plan-03\plan.json`

SHA-256:
`43dda454adf45352dee281680f51698ecf38d60731d33311a6ded139de167461`

The source commit is `77c2949c6d454ad81ba335f2a0b2a1601bbb8297`. An earlier
plan draft recorded an incorrect full commit ID and is superseded; its hash is
preserved. Acquisition attempt `perception-av-event-scale-media-01` was stopped
before its first ten-item manifest flush. Its partial files and an explicit
incomplete receipt are preserved, never admitted to the feature corpus. The
active acquisition uses a new output directory,
`perception-av-event-scale-media-02`.

## Planned comparison

Use the existing frozen DINOv2-small visual and CLAP audio encoders and the same
246K-parameter joint readout, with a parameter-matched video-only control.
Compare old training data (`fit` rows only from the existing 240-video / 525
event fit split) against old plus the new 233-video / 550-event fit split.
Each arm uses seed 17, the same 2,048 optimizer-update budget, the same new
58-video / 121-event selection split, the same 58-video / 146-event development
validation split, and a fixed 512-update validation interval. This estimates
the effect of increased unique video coverage at a matched update budget; it
does not isolate source distribution shift or prove natural AV necessity.

Best checkpoints are selected by selection NLL and kept separate from final
2,048-update checkpoints. Calibration and development-validation predictions
are saved once per arm. No previously scored 200-video examples are used for
checkpoint selection or training. The development-validation partition is
small and source annotations were previously parsed, so uncertainty and
annotation exposure will be stated with all results.

The existing scorer consumes only DINOv2 token 0. The scale extractor therefore
stores only that exact 384D candidate vector instead of all 257 tokens; the
loader accepts either representation and tests require exact parity. This
reduces feature-cache and training-host-memory costs without changing the
readout input. The fixed-update mode in `train_av_event_source.py` records
interval train CE (not a fixed-example generalization gap), validates at update
512/1,024/1,536/2,048, and keeps selected and final checkpoints separate.

## Execution gates and observations

- Plan creation used CPU only. Plan and acquisition validation tests pass.
- Acquisition attempt 01 failed closed on a manually transcribed existing
  holdout manifest hash. The hash was re-read from the actual file; no download
  began in that attempt.
- Acquisition attempt 02 initially reached 12 videos, then a TLS EOF failed a
  remote byte-range read. The 12 completed files and incomplete receipt were
  preserved. The same attempt resumed after validating all 12 file hashes;
  unflushed complete files were reconciled and transient TLS/connection errors
  now retry the identical pinned range with bounded exponential backoff.
  At the last observation it had reached 20/388 videos. Every row validates
  plan member length/CRC and cross-artifact media SHA-256.
- A CPU-only extraction preflight on 2026-10-21 verified all six local DINOv2 / CLAP
  artifacts against the hashes enforced by `_load_feature_models`; no model was
  loaded and no GPU was initialized:
  - DINOv2 weights `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`,
    config `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d`,
    processor `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`;
  - CLAP weights `d190141ffd119b9096d9f3e096b2285f8ce6fa401cdb666d6f53911a2517fee3`,
    config `f7a374a0dc4907b3a30a4e9964286055929c42d932d5abd0619f7b1a20438dd3`,
    frontend `9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753`.
  The local runtime reported Python 3.11.9, PyTorch 2.6.0+cu124, Transformers
  5.6.2, and PyAV 18.1.0.
- The full PTS audit passed: 388 videos, 908 events, no right-censored clips or
  events with partial/zero audio overlap. Audit SHA-256:
  `785c6c75648f9cd1a85cddbed868bf35235a803b9f1439ceb8a651c9081f53b5`.
- Crop joins and frozen feature extraction passed for all videos/events. The
  gated RTX 3080 extraction took 867.59 seconds; no Orbitune process was
  stopped. Feature result SHA-256 is
  `7907a617c97037fcabb001d46be70a2dc48c5487156c85d46b1463a3d8d0dd43`.
- Four CPU fits (joint/video-only × old-only/expanded fit coverage) each
  completed 2,048 updates, retained best and final checkpoints separately, and
  passed best-checkpoint reload checks. All used seed 17 and identical
  selection, calibration and development-validation IDs. Best steps were
  1,536, 2,048, 512 and 2,048 respectively.
- Paired video-cluster bootstrap (10,000 draws, seed 17) found intervals
  spanning zero for both Accuracy and NLL deltas in both readouts. Joint
  development Accuracy changed -0.7pp; video-only changed +4.8pp, with its
  Brier/ECE worsening. These are uncertain task-specific development results,
  not evidence of a general quality gain.
- The implementation tests and repository checks are being run for the
  recording commit. Full curves, metrics, artifact hashes, environment and
  limitations are in
  [`R2_PERCEPTION_AV_EVENT_SCALE_RESULTS_20261021.md`](R2_PERCEPTION_AV_EVENT_SCALE_RESULTS_20261021.md).

The prior fixed-checkpoint 200-video comparison remains available in
[`R2_PERCEPTION_AV_EVENT_FRESH_HOLDOUT_20261020.md`](R2_PERCEPTION_AV_EVENT_FRESH_HOLDOUT_20261020.md)
and is not reused as a new blind benchmark.
