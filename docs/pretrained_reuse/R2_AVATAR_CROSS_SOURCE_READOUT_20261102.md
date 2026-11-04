# AVATAR Cross-Source Audio-to-Video Readout (2026-11-02)

## Finding that invalidated the first choice set

The original frozen `option-choices-04` set is preserved, but its readout is not
valid evidence of AV category matching. In all 11,995 train and 1,693
validation rows, the positive video option used the same asset and exact
timestamp as the query audio. Timestamp equality alone uniquely identified the
correct option in 11,957/11,995 train rows and 1,689/1,693 validation rows.
The original metadata-only readout reached 99.41% validation accuracy. This is
a construction shortcut, not multimodal quality; do not use those metrics for
product claims or model selection.

## Corrected development task

Added opt-in `--cross-source-positive` to
`scripts/build_avatar_option_choices.py`. It leaves the original default and
its frozen artifact unchanged. In the opt-in task, the positive candidate has
the query's annotated category but comes from a different source-video group,
different asset, and different timestamp. All four candidate categories and
source groups are distinct, and no candidate source group is the audio query's
source group. Seed 17 controls deterministic selection and option order.

Rebuilding the corrected train/validation inputs and sidecars in memory with
the current generator reproduced all four saved file hashes exactly. Source
group bootstrap resamples query-audio groups, so repeated queries from one
source stay together.

The new output is
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02`.
It contains 11,995 train and 1,655 validation rows. Thirty-eight of the 1,693
validation queries lacked an eligible same-category candidate from another
source group and were retained as documented exclusions; there was no
resampling or audit access. Cross-split source-group and asset overlap are both
zero. The query source group and positive-option source group are stored
separately in the analysis sidecar and are excluded from model inputs.

The frozen feature caches were subset without re-encoding or changing their
source artifacts. Source hashes remain audio
`018bd96b433c5a56d26a6d26c101a5a5ff1a399c7e891efdcf5a681fc423c32c` and video
`129bfc5a40da3d8a96eee3657ea3874476c3777709f35071fe768f539ad1a3ff`. Subset
hashes are audio
`5736e2c7e094a93aea81fa13f517799292e790197dda6f6282140e3882636a58` and video
`6ab47dcc0f9c8fe691425fa45a6f1394c8520677829a419c51b93a45da068b6e`.

## Fixed readout

One seed-17 CPU readout was run with the existing frozen-feature MLP controls,
maximum 50 epochs, validation-NLL selection, patience 6, and `min_delta=0.001`.
It completed in 21.87 seconds. The selected joint checkpoint was epoch 4. No
GPU, cloud compute, sealed audit, or product promotion was used.

| Validation arm | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Uniform 4-choice | 25.00% | 1.3863 | 0.7500 | 0.0002 |
| Metadata-only | 25.44% | 1.3867 | 0.7502 | 0.0026 |
| Video-only | 31.18% | 1.3620 | 0.7408 | 0.0515 |
| Joint audio + video | 89.79% | 0.2493 | 0.1429 | 0.0231 |
| Joint, audio removed | 23.81% | 2.0333 | 0.9922 | 0.3589 |
| Joint, video removed | 25.02% | 1.3863 | 0.7500 | 0.0002 |
| Joint, audio shuffled across distinct assets | 24.23% | 3.7774 | 1.2271 | 0.5541 |
| Audio-category oracle control | 98.31% | 0.0471 | 0.0256 | 0.0067 |

The audio-category control is **not deployable**: its offline scorer is given
candidate category labels. It does not demonstrate audio-only option
selection. In this corrected task the metadata-only arm is near chance, and
the missing/shuffled-stream controls are consistent with the joint readout
using both frozen modalities. The paired joint-minus-video-only accuracy
difference is +58.61 percentage points (95% source-group bootstrap interval
+55.14 to +62.49; 2,000 replicates; seed 17; audio-query source-video group as
the resampling unit).

The joint training-to-validation NLL gap is 0.1991. Macro over 71 validation
categories, joint accuracy is 89.85%, NLL 0.2574, Brier 0.1462, and ECE-15
0.1013. These are development validation results for a narrow category-match
task, not native AVATAR localization, general video reasoning, 95–98% product
quality, or evidence that all modalities work. The fixed feature encoders
contain 1,961,239 audio and 114,365,184 video parameters; the learned joint
readout has 73,857 parameters.

## Reproduction receipts

- Source code revision used by the readout: `62b4e944ba5622605976e8071fdf97f4079675d`.
- Corrected train-input SHA-256: `0139f4f59a685c3f83890156b59c42d12aca9db54b22ee5c0a9c70b6ec7d60ef`.
- Corrected validation-input SHA-256: `c31a160d6262f065cb7df1ff20359a29b6444a307b6ebd0fcea9a3022932eba0`.
- Corrected train-sidecar SHA-256: `90adbf04450e0854fdb379dbcceb6aaa73790b205c08f1cf6c2cb786b6710a57`.
- Corrected validation-sidecar SHA-256: `e00d95134f016ec094dec59048084f429ef9b35d0ee76d3f7dbd45f203c8fc6b`.
- Choice-builder source SHA-256: `cae38ad15d7ebfa110376e5afcd3df371c1628bf7499d9458627d93b2576b8c7`.
- Readout source SHA-256: `338588c1311c19170d130ed2bf89db888627029e6b20e2a508f1dc8cc7bb536c`.
- Effective config SHA-256: `6f278780bef7c1edecd71e6a7f67ccd364eae97461f32eeb76a9588709ed1f00`.
- Predictions SHA-256: `e414a2595559c16702b2d758a49eab8c23d885db24a266024eb23f273ab463c7`.
- Report SHA-256: `bd7ec4c90f6afebf803d6ef85f3d0698b6324c4713a4c64fa34e5dd63dba4c69`.
- Best joint checkpoint SHA-256: `49cc9dd20fabf045960482e97d56bb5bec6d626cfc815b3834468700fd26d05e`.
- Full report and per-example predictions are in the output directory above,
  under `readout-02/`.

The corrected readout supports continuing this particular frozen-feature AV
diagnostic, while the large train/validation gap, 38 validation exclusions,
limited source family, and non-native task keep R2's broader quality gates open.
No shared Lens, merging experiment, or product promotion is justified by this
single validation run.
