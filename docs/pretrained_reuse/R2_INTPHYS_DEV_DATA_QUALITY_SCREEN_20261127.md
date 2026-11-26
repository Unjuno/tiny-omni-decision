# IntPhys development data quality screen — 2026-11-27

## Decision

Admit the downloaded IntPhys 2019 development payload as a **video-only,
task-specific temporal reasoning diagnostic**. It has explicit possible /
impossible labels and matched four-video scene groups. Do not use its labels to
fit model parameters: the benchmark authors reserve the development data for
selecting a plausibility score and comparing architectures. Any learned readout
must be fit from the official training partition. This screen is not evidence
for general multimodal Decision quality or the project's 95–98% aspiration.

## Pinned source and rights

- Official source: <https://intphys.cognitive-ml.fr/download.html>
- Component: `dev.tar.gz`, downloaded from the official publisher link only.
- Published license: CC BY 4.0. Cite Riochet et al., *IntPhys: A Framework and
  Benchmark for Visual Intuitive Physics Reasoning*.
- Published size/checksum: 3G / MD5
  `6a72715007f2f3b0e9546bfa8d8fc39b`.
- Retrieved archive: 3,203,229,667 bytes; MD5
  `6a72715007f2f3b0e9546bfa8d8fc39b`; SHA-256
  `441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a`.
- Local copy: `C:\CodexArtifacts\pretrained-reuse-r2-local\intphys-patha-data-quality-20261127-attempt-01\dev.tar.gz`.
  It is outside Git. No test archive was downloaded or opened. No training
  archive was downloaded.

The official starting-kit link returned a 211,712-byte archive whose MD5 does
not match the publisher's listed checksum (`cf699d546892d206c51ceff352a2534c`).
Its observed MD5 is `a1bc604b5b7fde93f5d38b57c09ab17c` and SHA-256 is
`e37aaf7da006b7657ae03f06c9ada88b9ac3f3b79bc88c7782390fa6d4c58569`. It is
retained outside Git as unverified and was not executed or used. The dataset
archive itself matches the published checksum.

## Observed data quality

`scripts/audit_intphys_dev.py` reads the compressed archive without extracting
media, validates its paths and metadata, checks every frame index, and hashes
the RGB-like `scene` frames to measure cross-label reuse. The full archive
passed this audit:

| Check | Observed |
|---|---:|
| Archive members | 108,360 |
| Video sequences | 360 |
| Videos per block | O1: 120; O2: 120; O3: 120 |
| Scene groups | 90 (30 per block) |
| Labels per block | 60 possible / 60 impossible |
| Renditions per block-scene group | 4, exactly 2 per label |
| Frames per sequence and channel | 100 |
| Channels | scene frames, depth, instance masks |
| Cross-label video pairs within matched groups | 360 |
| Equal RGB-frame hashes at the same frame index | 18,000 / 36,000 (50%) |
| Cross-label pairs sharing at least one identical RGB frame | 360 / 360 |

All four renditions from a `(block, scene-id)` group must remain together in any
internal split or resampling. The archive carries `header.is_possible` as a
boolean label for every video. These labels define a focused temporal
plausibility task, not arbitrary question answering. Static-frame overlap
across every opposite-label pair is useful evidence that the labels cannot be
explained solely by a unique per-video still frame; it does not rule out all
shortcuts or establish real-world generalization.

There are three image sequences, not audio or text. Treat depth and masks as
separate diagnostic inputs; do not silently count them as independent
modalities or as the project's audio/video/text/image suite.

The downloaded archive's observed count agrees with the official
`IntPhys-Baselines` README (360 dev videos total). The arXiv v3 paper's dev
section describes 360 videos per block; this count differs from the retrieved
archive and the baseline README. Use the hashed archive's observed counts for
this run record and retain the publication-count discrepancy as a source
version ambiguity rather than silently reconciling it.

The paper also states that all dev/test objects and textures appear in the
training set. This is therefore a benchmark diagnostic, not a novel-asset
generalization test. The held-out test has no labels and remains untouched.

## Evaluation boundary and next gate

- Development labels may select a fixed score or compare frozen feature
  readouts, but must not fit weights or calibration parameters.
- The training set is large (15,000 possible-only videos, split across four
  published archives of about 35G each). Before downloading any training
  archive, use the pinned VideoPrism runtime smoke below plus the existing R1
  license/processor records to estimate whether streaming a bounded official
  train shard into frozen feature extraction is viable on local hardware and
  storage.
- Keep all 4 variants of each scene group together. Report matched-group
  metrics and uncertainty; do not treat 360 videos as independent samples.
- Do not use the benchmark's hidden test or claim this result as a multimodal
  product pass.

## Pinned VideoPrism runtime smoke

The local checkpoint was rehashed on 2026-11-27: 457,475,296 bytes and SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, matching
the pinned R1 inventory. Its exact Hub revision is
`google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`;
the pinned source commit is `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, with
Apache-2.0 evidence recorded in the R1 inventory. The existing isolated
environment ran Python 3.11.16, JAX 0.6.2, Flax 0.11.1, and TensorFlow 2.18.1
on CPU; no packages were changed.

A one-video CPU smoke decoded evenly spaced scene frames 1, 15, 29, 43, 57,
71, 85, and 100 from `O1/01/1`, yielding input `[1,8,288,288,3]` and finite
frozen token features `[1,2048,768]`. Forward time was 7.71 s and total time
including process startup, hash checks, model load, and first JIT compile was
70.49 s. The sample's feature SHA-256 is
`574afede7e87282a1841c273082597ff3726fd92f5b5d05e4396062d9d2c3c41`; the
smoke script and result are retained outside Git beside the dev archive. No
labels were read, no weights were fitted, and no GPU task was started. This
validates media decode and the pinned CPU path for one sample; it is not a
throughput projection or task-quality result.

## Verification

- The audit CLI passed on the full downloaded archive with the publisher MD5.
- `tests/test_intphys_dev_audit.py`: 3 passed; covers valid matched labels,
  label imbalance rejection, and checksum mismatch rejection.
- Full CPU suite: 2,944 passed, 11 skipped.
- Ruff passed for the complete repository.
- The full data audit's machine-readable receipt is retained outside Git beside
  the payload; the one-video runtime smoke is also hash-recorded there.
- No model training, checkpoint creation, new GPU task, or test-set access was
  started in this data-screen stage.

References: [official IntPhys download and license page](https://intphys.cognitive-ml.fr/download.html),
[IntPhys paper](https://arxiv.org/abs/1803.07616),
[official baseline repository](https://github.com/rronan/IntPhys-Baselines).
