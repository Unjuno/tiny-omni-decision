# Frozen VideoSAUR train-only Decision result

The fixed [protocol](R2_VIDEOSAUR_TRAIN_DECISION_PROTOCOL_20261013.md)
completed on 2026-10-13. This is training-set learnability evidence, not validation
or general Omni quality. No checkpoint selection or sealed audit was performed.

## Execution and integrity

Source commit: `2a6b67c`. All 99 training videos / 48,094 consecutive pre-cut
frames were extracted with the frozen model. Extraction took 4,831.16 seconds;
peak CUDA allocation was 705,276,928 bytes, reserved 893,386,752 bytes.
All model weights remained exactly unchanged. The fixed loader admitted every
feature hash, source identity, frame count, sampling index and text routing.

The recorded protocol command ran successfully with exit code 0. Seed 17,
50 epochs, 650 optimizer updates, 198 unique questions / 99 videos and 9,900
question presentations; 115,008 trainable readout parameters. CPU fitting and
controls took 6.29 seconds. Final checkpoint reload predictions were exact.
All saved readout artifact hashes were independently rechecked. Existing
checkpoints and datasets were preserved. Local compute only; no cloud charge.

Outputs: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-slot-train-readout-01/`.
Extraction: `C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slots-01/`.
The [machine-readable report](R2_VIDEOSAUR_TRAIN_DECISION_RESULTS_20261013.json)
contains every epoch, controls, timings and artifact hashes. Config, ordered
examples, untrained/final weights and predictions remain in the output directory.

## Fixed final endpoint and controls

| Train input | Accuracy | NLL | Brier | ECE | Changed pairs both correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 75.76% | 0.443811 | 0.286884 | 0.064477 | 9/53 |
| Cyclic mismatched video | 35.86% | 2.436067 | 0.934279 | 0.430242 | 0/53 |
| Zero video | 23.74% | 1.105451 | 0.671325 | 0.114550 | 0/53 |

Normal initial/final Accuracy: 76.77% / 74.75%. All 46 unchanged-target
pairs are jointly correct, but only 9 of 53 changed-target pairs are jointly
correct. The model changes its answer between conditions on 13 of 99 videos.
The aggregate train result therefore hides a substantial conditional weakness.

| Epoch | Train Accuracy | Train NLL | Changed pairs both correct |
| --- | ---: | ---: | ---: |
| 0 | 25.76% | 1.099530 | 0/53 |
| 1 | 44.95% | 1.064877 | 0/53 |
| 10 | 63.64% | 0.783772 | 0/53 |
| 20 | 73.74% | 0.555329 | 2/53 |
| 30 | 73.23% | 0.497544 | 3/53 |
| 40 | 73.74% | 0.455124 | 3/53 |
| 50 | 75.76% | 0.443811 | 9/53 |

## Interpretation and next gate

The head uses video-dependent information to fit training examples, and exhibits
some conditional sensitivity. Mismatch degradation does not distinguish useful
reasoning from memorization. No validation gap or absence of overfitting can be
claimed. Falling train NLL also does not establish a representational ceiling.

This warrants a fixed independent development comparison of the saved endpoint,
without retraining or choosing an epoch from validation. Freeze its extraction
and comparison protocol before executing validation. Report normal, mismatched
and zero inputs, changed-target pairs and calibration on all existing validation
IDs. No model promotion, compression or quality target achievement follows from
this train-only result. Checkpoint redistribution and original config provenance
limitations from R1 remain unresolved.
