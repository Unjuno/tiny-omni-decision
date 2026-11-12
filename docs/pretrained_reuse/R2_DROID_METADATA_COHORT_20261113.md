# DROID 1.0.1 metadata cohort feasibility — 2026-11-13

## Decision

**Keep DROID `REVIEW_NOT_ADMITTED`; do not train or evaluate on this cohort.** The metadata sample contains enough *declared building/scene-ID tuples* to look promising, but these tuples are not independent physical scenes. The official DROID paper says scene IDs are generated when collectors indicate camera/robot movement, then manually consolidated by robot serial, lab, building, timestamps, and visual review; 2,080 raw IDs become 564 physical scenes. This metadata-only pass did not reproduce that manual mapping. It therefore cannot prove the 200 independent physical-scene groups required for a validation-quality gate.

Prior DROID 1.0.0 experiments also used part of this dataset and did not show video benefit over state-only on the later-shard task: video-only Accuracy was 39.42%, state-only 42.34%, and joint 39.42%; joint NLL (1.7790) was worse than state-only (1.4968). Those exposed development results are not retuned or re-evaluated here. A new task could differ, but no both-input-required task or fresh physical-scene cohort is established.

## Scope and pinning

- Repository source commit: `caf2035049f239cac608700c5d24c815d06ce9d4` (DROID source-screen PR #87 head).
- Source: Google Research DROID raw release prefix `robotics/droid_raw/1.0.1/`.
- GCS object inventory: 741,866 entries, 196,036,747 bytes, SHA-256 `af0a8141b88fae097873b2c83309e43bb326e5024d4ee63c9ef17c179f161c9d`.
- The inventory lists 74,896 metadata JSON objects and 8,528,498,975,639 total payload bytes. It was acquired before this cohort check; no video, HDF5, SVO, or NPY payload was fetched.
- Sample: 10,000 metadata JSON objects selected by deterministic SHA-256 ranking with seed 17. Sample-order/object-manifest SHA-256: `29488a82baebd9a3086b51a25952286133725a36875492f51108515eb5e3bdbb`.
- Each fetched JSON object was pinned to its GCS generation and checked against listed byte length and MD5. The selected metadata totaled 17,740,854 bytes (about 16.9 MiB). The request/receipt interval was approximately 00:08–00:30 JST; no GPU was used.
- External receipt directory: `D:\CodexArtifacts\pretrained-reuse-r2\droid-metadata-cohort-20261113-attempt-07`. Metadata receipt SHA-256: `05819035bdec7b0a3b561ac86bdbf45ee5d3130a7ff12ae771c8a0943ad5dea5`; summary SHA-256: `80fd1b5cdade6424848c4198a14d1967fdef398b652829f51a3377ff1beebc5f`.

The parser selected `building`, `scene_id`, collector ID, HDF5 path, and MP4 path fields. The JSON objects also contain task/success fields; their values were not selected, used, logged, or persisted. This was a metadata-only source-feasibility check, not a task-label or model evaluation.

## Observed counts

| Check | Result |
|---|---:|
| Metadata JSON objects in pinned inventory | 74,896 |
| Fixed-seed metadata sample | 10,000 |
| Distinct `building + scene_id` tuples in sample | 1,586 |
| Distinct raw `building` strings in sample | 61 |
| Distinct collector IDs in sample | 78 |
| Physical scene groups verified | **UNKNOWN** |
| Video path references / found in the pinned inventory | 50,000 / 49,189 |
| Distinct video path references in sample | 29,988 |
| Repeated video aliases within a metadata record | 20,000 |
| Cross-record duplicate video path occurrences | 12 |
| HDF5 path references found in the inventory | 9,990 / 10,000 |
| Video/HDF5/SVO/NPY payload files downloaded | 0 |

The sampled raw identifier counts are not asserted to match paper-level physical scenes, buildings, or collectors. The 61 building strings and 78 collector IDs are not reconciled to the paper's 52 buildings and 50 collectors; version/scope and identifier normalization remain unknown. Missing file references also mean a future cohort would need a completeness gate before any feature extraction.

## Group-integrity finding

The official paper's Appendix D defines a unique scene as a substantial change to the robot's workspace. A raw `scene_id` is created when the collector indicates that the robot or external cameras moved. The authors report 2,080 such IDs, then merge repeated physical scenes using robot serial, collecting lab, building, timestamp order, and visual review. The official schema exposes those raw fields but does not provide the paper's final scene-equivalence mapping.

This audit only counted `building + scene_id` tuples. It did not load or inspect camera frames, use HDF5 values, or reconstruct the manual scene mapping. These tuples cannot safely be split across train/validation/audit as if each were an independent physical scene. Building-only grouping yields at most the paper's 52 buildings, below the 200-group validation floor; scene-ID grouping risks leakage between repeated physical workspaces. Therefore no split was frozen and no task cohort was admitted.

## Existing-exposure and task-quality limits

The earlier DROID next-action work used official 1.0.0 shards 00000–00019 and already observed a four-site validation. The later-shard transfer used 137 episodes from the same 14-site set, with no video advantage over sensor-only. This new 1.0.1 metadata sample was **not** cross-walked to those exposed episodes or sites. It does not establish any fresh evaluation cohort.

References:
- [Official DROID paper, Appendix D](https://arxiv.org/html/2403.12945#appendix.D): scene-ID creation and manual physical-scene deduplication.
- [Official DROID schema](https://github.com/droid-dataset/droid/blob/main/droid/postprocessing/schema.py): raw `building`, `scene_id`, robot, lab, and camera-path fields.
- [Prior later-shard frozen transfer report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-droid-temporal-transfer-20261028/docs/pretrained_reuse/R2_DROID_LATER_SHARD_FROZEN_TRANSFER_RESULTS_20261028.md): previously observed 1.0.0 development comparison.

## Preservation and next gate

- No training, inference, checkpoint selection, or sealed-audit access.
- No video, trajectory, SVO, or image bytes were downloaded or opened.
- No existing DROID or Teacher artifact was changed; all sample receipts are in the new external attempt directory and are excluded from Git.
- No paid/cloud compute or GPU time was used.
- Keep DROID out of model fitting unless a publisher-backed physical-scene mapping (or a defensible low-cost dedup method with independent review) and an exposure crosswalk to the prior 1.0.0 cohort are available. Do not infer independence from the 1,586 raw metadata tuples.
- R2 M-2 remains open. This source result does not justify shared-Lens/R3, model merging, or a quality claim.