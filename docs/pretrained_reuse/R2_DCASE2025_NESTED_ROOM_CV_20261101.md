# DCASE2025 nested room cross-validation

## Scope

This is a development-only follow-up to the one-shot DCASE2025 public dev-test
evaluation. It uses official dev-train frozen features only and estimates
whether validation-based early stopping improves transfer to unseen rooms.
It does not access dev-test or sealed-audit labels, and it does not establish
product quality or a general multimodal capability.

## Protocol and provenance

- Source branch commit: `d72d40aa29a73cb98f65ff8e10bc74a0b22d69cc`.
- Feature corpus: `C:\\CodexArtifacts\\pretrained-reuse-r2\\dcase2025-av-features-20261031-attempt-01`.
- Feature preflight SHA-256: `1be433e7f94a997e30b37caab89f74ee77477528c61437f61ce68d6917458524`.
- Effective config SHA-256: `c6636730cdd3c2dbf46a18273ce752d3e74b7b8ecbb0e983400c320c9e28ae23`.
- OOF result SHA-256: `d15336a72b62067f31d0fd94fbb61f681c8c7c945dbf161fc99afed38e4a5758`.
- OOF predictions SHA-256: `92e52f699cbf23d0099643f204c9b8a447f9275c82b0f0db450925a6a7426b7e`.
- Paired bootstrap SHA-256: `e16974e76ec6c42c2baeef4b2b18adddc5556fefe45f41c8b2f083e0130b2e3e`.
- Runner SHA-256 at run: `2136f2096303c22bd664fc20dfd1b3cdcea961360f7a2810aa13f5a5d3aaa28e`.
- Readout module SHA-256: `ef4a020e28fd0c49f94c4c7955d5694b584d7e8d3e67a5e85e933836005c69a0`.
- Command: `python scripts/run_dcase2025_nested_room_cv.py --features C:\\CodexArtifacts\\pretrained-reuse-r2\\dcase2025-av-features-20261031-attempt-01 --output C:\\CodexArtifacts\\pretrained-reuse-r2\\dcase2025-nested-room-cv-20261101-attempt-01 --seed 17 --max-epochs 30 --patience 5` with `PYTHONPATH=.;src` and `CUDA_VISIBLE_DEVICES=''`.
- Seed 17. Outer leave-one-room-out across all 9 dev-train rooms; each sample is outer-tested once. Each fold selects one different inner validation room, leaving 7 rooms for training. Parent recordings are checked group-disjoint across train, validation and outer test. Standardization uses inner-train statistics only.
- Existing readout architecture, width 64; frozen encoders and features; AdamW (`lr=0.003`, `weight_decay=0.0001`), batch size 256, at most 30 epochs, patience 5, minimum validation-NLL improvement 0.0001. Best epoch selected by inner-validation NLL. No outer-test selection.
- 15,485 examples, 78 parent recordings, 9 rooms; labels: 11,682 offscreen / 3,803 onscreen. Run completed in 40.95 seconds on CPU at `2026-11-01T00:46:13Z`.
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6. The RTX 3080 was occupied by an unrelated Orbitune process; this experiment used CPU and did not interfere with it.

Two launches failed before model fitting or output creation: first, the package root was missing from `PYTHONPATH`; second, the old feature preflight lacked a newer split-identity field. The runner now checks its frozen dev-train-only protocol and ID-order hash instead. No corpus or labels were changed.

## Out-of-fold results

Metrics are pooled over outer held-out predictions. ECE uses 15 fixed bins.

| Arm | Accuracy | Balanced accuracy | NLL | Brier | ECE-15 | Onscreen recall |
|---|---:|---:|---:|---:|---:|---:|
| Event-prior baseline | 75.44% | 50.00% | 0.5493 | 0.3656 | 0.0097 | 0.00% |
| Audio | 74.94% | 50.01% | 0.5793 | 0.3871 | 0.0692 | 1.03% |
| Video | 73.01% | 57.21% | 0.5513 | 0.3607 | 0.0598 | 26.16% |
| Joint | 73.77% | 60.61% | 0.5418 | 0.3552 | 0.0387 | 34.74% |

Room-macro results (accuracy / NLL / Brier / ECE-15): event prior
75.13% / 0.5548 / 0.3699 / 0.0257; audio 74.68% / 0.5808 / 0.3892 /
0.0718; video 74.39% / 0.5277 / 0.3438 / 0.0627; joint 74.89% / 0.5231 /
0.3406 / 0.0609.

Mean selected epoch was 1.11 for audio, 2.11 for video and 1.44 for joint.
Train-to-inner-validation NLL gaps were positive in 8/9 audio folds and all
video and joint folds; mean gaps were 0.0781, 0.1278 and 0.2122 respectively.
This is consistent with rapid overfitting, especially for joint readout.

## Paired uncertainty diagnostic

Posthoc paired cluster bootstrap used the saved OOF predictions only (10,000
replicates, seed 17). No model fitting or dev-test evaluation occurred.
Differences below are joint minus comparator; intervals are 95th percentile
intervals.

| Cluster | Comparison | Accuracy difference (95% CI) | NLL difference (95% CI) | Brier difference (95% CI) |
|---|---|---:|---:|---:|
| Parent recording (78) | Audio | -0.0116 [-0.0345, 0.0087] | -0.0374 [-0.0641, -0.0095] | -0.0319 [-0.0521, -0.0106] |
| Parent recording (78) | Video | +0.0077 [-0.0057, 0.0217] | -0.0095 [-0.0245, 0.0049] | -0.0055 [-0.0167, 0.0052] |
| Room (9) | Audio | -0.0116 [-0.0694, 0.0300] | -0.0374 [-0.0889, 0.0212] | -0.0319 [-0.0738, 0.0190] |
| Room (9) | Video | +0.0077 [-0.0068, 0.0228] | -0.0095 [-0.0280, 0.0091] | -0.0055 [-0.0198, 0.0063] |

Only nine room groups make room-level intervals low-power. The parent and room
bootstrap have the same point difference because the pooled paired sample
metric is unchanged; their uncertainty differs. This is development evidence,
not a blind confirmatory result.

## Interpretation

Early stopping on an inner room reduced the severity of the earlier all-data
30-epoch fit but did not beat the event-prior accuracy baseline on outer-room
OOF predictions. The joint arm improved balanced accuracy and pooled proper
scores over video-only, but its raw accuracy remained below the event prior.
The one-shot dev-test evaluation reported in
[`R2_DCASE2025_AV_READOUT_20261101.md`](R2_DCASE2025_AV_READOUT_20261101.md)
showed a larger failure under held-out rooms. The discrepancy points to strong
room/domain variation and/or instability from only nine room groups; it does
not prove an architecture ceiling or that learning quantity is the sole cause.

Do not promote this readout, tune it against dev-test, or claim that the task
requires both modalities. Preserve the public dev-test result as the one-shot
reference. Further work should first secure an independently admitted task
with group-disjoint validation and meaningful both-input-required labels.
Existing v1/v2 model artifacts, checkpoints, features, and corpora were not
modified; no sealed audit was opened; no GPU training was started.
