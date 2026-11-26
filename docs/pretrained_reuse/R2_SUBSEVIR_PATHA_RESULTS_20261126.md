# sub-SEVIR frozen-feature Path A result — 2026-11-26

## Result

This is a measured, task-specific sensor Decision experiment. It is not a
general video, image, sensor-fusion, or Omni quality result.

The selected two-stream DINOv2 + LSTM readout reached **89.62% Accuracy** on
the fixed internal validation split of 3,988 windows / 998 storm-event groups
(NLL 0.2380, multiclass Brier 0.1485, ECE-15 0.0194, mean confidence 0.9103).
It misses the aspirational 95–98% target. The publisher validation/test and any
sealed audit remain unopened.

| Frozen-feature readout | Selected epoch | Accuracy (95% storm-group bootstrap CI) | NLL (95% CI) | Brier (95% CI) | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Satellite only | 2 | 85.58% [83.69, 87.44] | 0.3171 [0.2847, 0.3511] | 0.2020 [0.1789, 0.2260] | 0.0257 | 0.8778 |
| Radar only | 5 | 88.97% [87.28, 90.63] | 0.2453 [0.2148, 0.2768] | 0.1557 [0.1345, 0.1773] | 0.0259 | 0.9134 |
| Satellite + radar | 2 | **89.62% [87.95, 91.22]** | **0.2380 [0.2087, 0.2687]** | **0.1485 [0.1280, 0.1696]** | **0.0194** | 0.9103 |

Intervals use 10,000 storm-event bootstrap replicates, seed 17. Paired
joint-minus-satellite differences favor joint: Accuracy +2.31 to +5.82
percentage points; NLL -0.1065 to -0.0522; Brier -0.0732 to -0.0345. Against
radar-only, paired Accuracy difference is -0.78 to +2.05 points and NLL
difference is -0.0298 to +0.0150; both intervals include zero. Thus adding
satellite input has **not** demonstrated an improvement over the best
unimodal candidate. Do not claim that this task requires two inputs.

Uniform probabilities give NLL 0.6931 / Brier 0.5. Training prevalence is
50.66% positive; the majority-class validation accuracy is 49.37%. Raw
softmax probabilities are not post-hoc calibrated. ECE-15 is reported, but no
ECE bootstrap interval or calibration threshold pass is claimed.

## Learning curve and overfitting

All candidates used seed 17, all 15,828 train windows, batch 128, AdamW
`1e-3` / weight decay `1e-4`, gradient clip 1.0, at most 60 epochs and the
frozen selector `minimum validation NLL`, `min_delta=1e-4`, patience 10.

| Candidate | Epoch 1 rolling train Acc / validation Acc / NLL | Selected epoch validation Acc / NLL | Last epoch rolling train Acc / validation Acc / NLL | Epochs run |
|---|---|---|---|---:|
| Satellite | 83.55% / 85.18% / 0.3363 | 85.58% / 0.3171 | 96.47% / 85.91% / 0.4783 | 12 |
| Radar | 87.09% / 87.89% / 0.2720 | 88.97% / 0.2453 | 96.51% / 86.58% / 0.3970 | 15 |
| Joint | 88.46% / 89.34% / 0.2400 | 89.62% / 0.2380 | 97.51% / 89.07% / 0.3601 | 12 |

All three show overfitting: rolling train accuracy continues to rise while
validation NLL worsens after the early selected epoch. Rolling train metrics
are optimizer-fit summaries, not fixed-example generalization estimates.
Best and final checkpoints are separate files, and a post-run reload audit
reproduced every validation probability exactly (maximum delta 0.0).

## Input and time controls

On the selected joint checkpoint, intact inputs scored 89.62% / 0.2380 NLL.
Zeroing satellite scored 86.46% / 0.3221; zeroing radar scored 78.99% / 0.4478.
An event-level radar shuffle, matched by the available forecast horizons,
scored 61.74% / 0.9549. Paired group-bootstrap Accuracy difference for shuffled
minus intact was [-31.04, -24.77] points; NLL difference was [+0.640, +0.795].
This says the joint readout uses event-associated radar features; it does not
prove that radar improves on the radar-only model.

Time reversal scored 89.19% / 0.2482 NLL. Its paired Accuracy interval versus
intact spans zero [-1.18, +0.33] points; its NLL difference is small
(+0.0011 to +0.0196). This provides weak evidence that the tested ordered
sequence helps proper scoring, not strong evidence of temporal reasoning.

## Integrity, source, and local runtime

- Official source: [sub-SEVIR Zenodo record](https://zenodo.org/records/7011372),
  version 0.1, CC BY 4.0. The locally verified archive is 1,974,264,123 bytes,
  MD5 `94ae916d24224500c2f531bb61440dab`, SHA-256
  `3c8f89da9f28485dfa0b47f6f3394abe92cbf9be02f5c7bb6ca2deb2d6d05856`.
- Only the publisher training Zarr was extracted. Its `.zmetadata` SHA-256 is
  `cdcef87021e848a08f661e7caa30abb7a31810d7326f7c9a286c468d0523101a`.
  The project-created forecast split is group-disjoint: 3,969 train and 998
  validation events; 15,828 and 3,988 windows, respectively.
- Split audit reports zero event overlap, zero exact raw-frame overlap,
  zero raw duplicate frames, zero full two-stream normalized eight-frame
  sequence overlap, and zero feature-cache event-membership errors.
- After p01/p99 pseudo-RGB normalization, 949 single radar-frame hashes and
  40 radar-only eight-frame input sequences recur across splits. These are
  repeated no-signal radar encodings, not repeated raw pixels or event IDs;
  the satellite stream differs and the combined two-stream sequence hashes do
  not overlap. This limits interpretation of the radar-only baseline and is
  retained without filtering examples.
- Pinned encoder: `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`,
  Apache-2.0 per its [pinned R1 audit](R1_DINOV2_SMALL_20261012.md); 22,056,576
  frozen FP32 parameters, checkpoint SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- Hardware: local RTX 3080 Laptop 16 GB, PyTorch 2.6.0+cu124,
  Transformers 5.6.2, FP32. Ten-minute extraction dry run: 62,560 stream frames
  / 600.17 s / 104.24 frames per second / 292,862,464 peak allocated VRAM.
  Full extraction: 54,585 unique source frames, two streams, 1,040.56 s,
  292,862,464 peak allocated VRAM. Three readout fits took 29.91 s and peaked
  at 63,988,736 allocated VRAM. Cloud compute and paid compute: zero.
- Readout parameter counts: 215,650 trainable for each unimodal scorer and
  431,202 for joint. Joint total including frozen DINOv2: 22,487,778
  parameters. The best joint readout is 1,725,848 bytes. This is a sensor-task
  submodel, not the full product assembly; weights are unquantized, and no
  deployed package or device latency was measured.

## Decision and limits

This Path A candidate is useful evidence that a small frozen visual encoder
plus a temporal LSTM can score an adapted satellite/radar forecast task at
89.62% validation Accuracy. It is below 95%, is selected on a repeatedly used
development validation, and is specific to this dataset and mapping. The
radar-only candidate is statistically tied with joint under the paired event
bootstrap; temporal-order impact is modest. The sensor fusion gate therefore
remains **unproven**. No VideoPrism bulk features were extracted; its CPU-only
batch-8 runtime projected roughly 17 hours for all paired windows, so it was
deferred before model selection.

This result neither advances nor rejects BERT for text Decision tasks. The
repository already records Japanese BERT-base at 81.65% on the selected
MASSIVE development split and its separate five-choice probe; those results
support a task-specific text-teacher/reference, not a language or multimodal
quality pass. Further BERT work should use a new, source-disjoint text task and
fixed split; it does not address this sensor/video temporal pipeline.

All run files remain outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-patha-dinov2-lstm-v1`;
manifest, feature cache, throughput, split audit, checkpoint reload audit and
predictions are in the same active artifact directory. No legacy worktree,
checkpoint, model, corpus or publisher sealed audit was modified or opened.
