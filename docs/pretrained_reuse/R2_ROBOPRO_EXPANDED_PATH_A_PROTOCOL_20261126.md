# RoboPro expanded Path A protocol — 2026-11-26

## Purpose and status

This is one fixed-size follow-up to the 16-prompt-group development probe in
[`R2_ROBOPRO_PATH_A_ACTION_READOUT_20261126.md`](R2_ROBOPRO_PATH_A_ACTION_READOUT_20261126.md).
The earlier result favored Text+State over Text+State+Video-GRU, but had only
16 development prompt groups and therefore did not support a stable modality
conclusion. This run uses previously unselected exact prompt groups from the
same pinned publisher TRAIN release. It is still development evidence, not a
publisher test or blind audit.

**Status: frozen sample downloaded and audited; CPU VideoPrism extraction is in
progress. No model-quality result is available yet.**

## Frozen sample and exclusions

- Dataset: `Hoshipu/RoboPro`, revision
  `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0` (publisher-declared CC BY 4.0).
- Seed: 17; 64 exact prompt groups per partition; four episodes per group.
- 256 train episodes / 64 prompt groups and 256 development episodes / 64
  prompt groups; three causal query points per episode.
- Excludes all episode IDs and exact prompts used by the earlier 16-group
  sample, plus its eight documented smoke episodes.
- Sample manifest:
  `manifests/pretrained_reuse/robopro-action-probe-expanded-sample-20261126.json`
  (SHA-256
  `6c78afea26b2c1862b3b6a75b10cf761df2a723456717afd3e1da54369a4039a`).
- Effective config:
  `configs/pretrained_reuse/robopro_action_probe_expanded_v1.json`
  (SHA-256
  `df022cbb20961c2470d67361b73e51c48aa3268964f59c8b3885c4cb1028193f`).
- Frozen feature encoders, readout arms, optimization, seed, query timing, and
  metric selection match the earlier fixed protocol. This is a sample-size
  expansion only; no encoder, arm, hyperparameter, or label definition changed.

## Data and integrity audit

The 512 episodes were acquired from the pinned revision into the separate local
attempt `C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-expanded-v1-attempt-01`.
The download receipt binds the sample SHA and verifies every downloaded file;
receipt SHA-256 is
`dc7241dafa14aff8c3716a58c1aa5d7fb6715cd1c49b28c35708023785e72a61`.
The download contains 1,024 files totaling 596,960,654 bytes. Parquet extraction
checked 126,586 rows and produced a 4,216,638-byte state/action sidecar with
SHA-256
`7f3945660614a74cc536292431b2518e872d10b2b006244e4500de3801745056`.

The full media/split audit passed:

- 512 episodes, 384 four-choice candidate queries, zero duplicate candidate
  queries;
- zero exact-prompt overlap, zero file-hash overlap, and zero normalized
  32×32-frame overlap across train/development.

Audit report is external at
`C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-expanded-v1-attempt-01\data-audit.json`.
Physical scene identity is not exposed by the source metadata, so scene-level
independence remains unknown. Recorded demonstrator actions are retrieval labels,
not verified success, safety, or unique optimality labels.

## Runtime and execution

Text features for 128 prompts were extracted on CPU in 7.03 seconds with the
same pinned MiniLM-L3 checkpoint. VideoPrism remains CPU-only; no GPU or cloud
compute is used. The first extraction attempt completed model inference but
hit the old extractor's hard-coded `(128, 3, 8, 768)` shape check before saving
the cache. A sample-size-derived shape validator and regression tests were
added. The corrected extraction is running in the same new attempt directory.

Current fitted readout result, timings, hashes, software versions, and measured
peak memory will be appended only after a successful extraction and one fixed
readout fit. No official test, sealed audit, product promotion, or quantization
is part of this experiment.
