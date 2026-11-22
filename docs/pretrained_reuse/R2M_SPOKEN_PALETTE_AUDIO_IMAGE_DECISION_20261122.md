# Spoken-color screen-action Decision pilot

**Status:** executed controlled development pilot; not product-quality evidence.

## Purpose and scope

This pilot asks one narrow question: can a small shared Decision readout use
both a spoken command and a visual screen to choose one supplied action? It
does not measure general audio, image, text, video, temporal reasoning, or
sensor quality. The four-choice task is deliberately controlled: select the
screen position whose colored tile matches the spoken color. Screens are
generated, contain no words, and show all four colors. The four color labels
are the only task vocabulary.

The source is the explicitly approved CC BY 4.0 Lombard GRID TRAIN partition.
Only its 4,080 approved TRAIN utterances were used. No source validation/test
partition, sealed audit, paid compute, or GPU was used. Synthetic screens use
12 even-permutation layouts for fitting and 12 odd-permutation layouts for
selection/calibration/evaluation. Speaker groups are disjoint across the four
partitions. Across the factorial task, audio-only and image-only have a 25%
macro-accuracy ceiling; both inputs identify one unique action. The frozen
corpus preflight reports no cross-split audio-source, normalized-content, or
rendered-screen-hash overlap.

## Pinned implementation and inputs

- Source branch at run: `codex/pretrained-reuse-next-20261115`.
- Git HEAD at run: `2db54c9b74f3548083983ce4d505d2736918e92c`; the pilot code and
  config were then uncommitted. The source scripts and config are included in
  the follow-up commit. After training, only unused helper/constant removal,
  import sorting, docs, and tests changed; no training behavior changed.
- Final source SHA-256: trainer
  `f429e0762942449bb1734c6f21cd81052633b7c385a8864214f8d9817596cc56`, corpus
  builder `b5d068713f120c1d561a20dcbc003465a46c4b87b55f334b0ad9439b4de47565`,
  task generator
  `171e645a83001f122fb966db0df4e59624552a0e4fc601c5ed59c324774c286b`.
- Run-time config bytes: SHA-256
  `d6fe92c1a702fef8dc20f4f25c1138e4079dcef022c9168e15134035db771996`.
  Final checked-in `configs/pretrained_reuse/spoken_palette_decision_v1.json`
  hash: `316e86897a6e6768feb6c730f101a8bb9fc66a3c4f0dbc7d15a8303df714b775`.
  A final review found the run-time JSON repeated the same
  `selector_min_delta: 0.0001` key twice. The duplicate was removed after the
  run; both parsed values were identical, and the effective training setting
  remains unchanged. The original run-time bytes are preserved by hash and in
  the external `effective-config.json`.
- Corpus manifest: SHA-256
  `b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a`.
- Lombard source manifest: SHA-256
  `5876e57a4c71108fe3483fde6e18f3977121942c9208567488f042247626577d`;
  frozen original TRAIN rows: SHA-256
  `259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0`.
- Whisper-tiny pooled audio bank (384D): SHA-256
  `e09aab88f31957c53c2e48f38a7e79971739efc916547cd5fd3dd6a1bca0a509`,
  pinned encoder revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
- DINOv2-small pinned revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`; loaded model file SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- Both encoders were frozen. The shared option scorer trained 98,561
  parameters. Counted resident model total before quantization was
  30,363,521 parameters (8,208,384 Whisper + 22,056,576 DINOv2 + 98,561
  scorer). No quantization was performed. The two pretrained encoder files
  are not included in the readout checkpoint; a distributable package size
  was not measured.

## Data and training

| Partition | Speaker groups | Unique utterances | Decision rows | Unique screens |
|---|---:|---:|---:|---:|
| Train | 24 | 2,694 | 64,656 | 24 |
| Selection | 4 | 394 | 9,456 | 24 |
| Calibration | 4 | 595 | 14,280 | 24 |
| Evaluation | 4 | 397 | 9,528 | 24 |

The fixed seed-17 configuration used AdamW, batch 256, maximum 40 epochs,
cosine schedule, and selection macro-color NLL with `min_delta=0.0001` and
patience 7. It stopped after 12 epochs when selection patience was exhausted.
The selected checkpoint is epoch 5; the final epoch-12 checkpoint is retained
separately. The scalar temperature `0.9012504626` was selected from the
predeclared 81-point `[0.25, 4.0]` grid using calibration data only. Evaluation
rows were opened only after candidate and temperature freeze.

Training history (selected metric is macro-color NLL):

| Epoch | Updates | Train fixed macro Accuracy | Train fixed macro NLL | Selection macro Accuracy | Selection macro NLL |
|---:|---:|---:|---:|---:|---:|
| 1 | 253 | 1.0000 | 0.00177 | 0.9957 | 0.01460 |
| 2 | 506 | 1.0000 | 0.00056 | 0.9952 | 0.01441 |
| 3 | 759 | 1.0000 | 0.00028 | 0.9952 | 0.01415 |
| 4 | 1,012 | 1.0000 | 0.00017 | 0.9953 | 0.01403 |
| 5 (best) | 1,265 | 1.0000 | 0.00011 | 0.9952 | 0.01373 |
| 8 | 2,024 | 1.0000 | 0.00005 | 0.9952 | 0.01454 |
| 12 (final) | 3,036 | 1.0000 | 0.00002 | 0.9952 | 0.01497 |

The train-selection gap in NLL widens after epoch 5 while selection NLL no
longer improves; this is mild overfitting in calibration-sensitive loss even
though Accuracy remains nearly unchanged. The epoch-5 checkpoint was kept.

## Evaluation

All metrics below use the same 9,528 evaluation decisions, the same frozen
candidate, and the one calibration temperature. Accuracy is also macro
averaged across the four spoken colors.

| Condition | Accuracy | Macro-color Accuracy | NLL | Macro-color NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|
| Matched speech + screen | 0.99990 | 0.99989 | 0.00119 | 0.00120 | 0.00041 | 0.00086 |
| Audio replaced by TRAIN mean | 0.24937 | 0.25000 | 1.71762 | 1.72500 | 0.88689 | 0.27305 |
| Visual content replaced by TRAIN mean; coordinates kept | 0.25000 | 0.25000 | 1.38700 | 1.38700 | 0.75035 | 0.01160 |
| Wrong-color audio donor | 0.00000 | 0.00000 | 13.39456 | 13.39939 | 1.99753 | 0.99883 |
| Wrong-layout screen donor | 0.15575 | 0.15625 | 11.18449 | 11.19005 | 1.68683 | 0.84336 |

Matched accuracy by color was blue 1.0000, green 1.0000, red 1.0000, and white
0.99957. Replacing either modality's content returns performance to chance;
crossing the audio and screen also sharply damages decisions. These controls
support paired-input use in this constructed task. The near-perfect score is
not evidence of broad multimodal generalization: there are only four colors,
four evaluation speaker groups, generated screens, and one preselected task.
The evaluation is development-only and influenced future research choices.

The shared scorer's probability vector changed by at most `1.2e-7` after
permuting candidate order and undoing that permutation. A typed choice JSON
example is preserved in the external result artifact.

## Runtime and artifacts

Run started `2026-11-22T05:58:27Z` and finished
`2026-11-22T06:00:01Z`. CPU-only, four Torch threads, Python 3.11.9,
PyTorch 2.6.0+cu124, NumPy 2.4.6, Windows 10 build 26300. Fit plus evaluation
took 93.45 seconds; DINO screen feature extraction took 4.20 seconds before
fit and 1.77 seconds after freeze. Peak sampled RSS was 1,120,161,792 bytes.
CUDA use and cloud cost were both zero.

External artifacts are under
`C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-fit-attempt-01`.
Best readout checkpoint SHA-256 is
`48b4a12401f9da88472b290536ca5e3fe3b708319af9d221caa5ce4b75e9c1b5`; final
checkpoint SHA-256 is
`8b617550dbc32b8a21f89daf5796d4567ce4c77bf924aed251481a082b4b42fe`.
Training history SHA-256 is
`077c8ec01f03dcfd0784ccf649f3d10575d2734c46e048a7e8170eb5e01334e9`; result
JSON SHA-256 is
`ff1c817b870a24e1b7616c845e17188e01937e355e04f761255f41d038406085`;
prediction JSONL SHA-256 is
`f46ceefd1f4c2a249e66c96bb2c5c3826d2faeb91ef3ded4653514f3245b38b7`.
Model weights, feature arrays, and per-example predictions remain external
and are not added to Git.

## Decision

This pilot demonstrates a small, parameter-efficient, typed supplied-option
Decision mechanism can combine audio and image features for a narrow task.
It does **not** complete the Pretrained World Representation Reuse goal,
validate four modalities, establish natural video use, meet a general 95–98%
accuracy target, or justify product promotion. The global objective still
needs independent task-specific datasets and evidence for text, natural
images, audio, and video, a package/runtime audit, and later quantization
measurements. Keep this as a mechanism pilot and do not treat its development
evaluation as a blind benchmark.
