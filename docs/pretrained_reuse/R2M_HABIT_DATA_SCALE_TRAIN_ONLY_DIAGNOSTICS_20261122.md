# HABIT scale-up: train-only grouped OOF diagnostics — 2026-11-22

## Finding

The expanded HABIT train package does not show a reliable gain from the
video+state readout over either single stream. On 606 train transitions,
five-fold grouped out-of-fold (OOF) predictions were generated for the fixed
text-only, video-only, state-only and video+state readouts at epochs 1, 2, 4
and 8. The 37 connected task/video components were never split across folds.
The development-validation split and sealed audit were not used; no checkpoint
was selected.

Train batch cross-entropy declined in every arm while OOF NLL generally rose
after epoch 1. At epoch 8, paired video+state had 38.1% Accuracy, 1.141 NLL,
0.671 Brier and 0.164 ECE, versus 42.7%, 1.078, 0.632 and 0.101 for
state-only. A paired bootstrap over 37 connected components (10,000 draws,
seed 1701) estimated video+state minus state-only Accuracy at -4.62 points
(95% percentile interval -9.08 to -0.85), NLL at +0.064 (0.007 to 0.129),
and Brier at +0.040 (0.005 to 0.085). The corresponding video-only contrast
was less conclusive: Accuracy -5.28 points (-12.32 to +0.96), NLL -0.017
(-0.142 to +0.085), and Brier +0.007 (-0.048 to +0.059).

This is evidence of poor generalization/overfitting within this small
single-source development task, not a general architecture ceiling. The
epoch-8 fusion comparison is exploratory across fixed observation epochs;
there was no checkpoint selection. Do not add epochs or seeds to this same
pilot. A different, adequately grouped task/source is needed before another
fit.

## Protocol and provenance

- Experiment: `habit-train-only-grouped-oof-seed17-data-scale-v1`.
- Seed 17, CPU, 5 folds, 8 epochs per fold and arm, batch size 32, AdamW,
  learning rate 0.001, weight decay 0.0001, cross-entropy, randomized option
  order and fold-train-only sensor normalization.
- OOF observations were fixed in advance at epochs 1/2/4/8. No early stopping,
  checkpoint selection, hyperparameter search or validation metric computation.
- Data: frozen scale-up train rows only; 606 examples, 37 task groups, 198
  unique videos and 37 connected components. The package contains 228
  development-validation rows and zero sealed-audit rows; neither split was
  loaded by the fit or OOF analysis.
- Folds held out 120–123 examples and 7–8 task groups (38–43 videos). Each
  train example received exactly one held-out prediction per arm and epoch.
  Training consumed 77,568 row presentations over 2,560 optimizer updates.
- The data package's `sealed_audit_accessed`, `action_columns_read` and
  `training_performed` provenance fields were checked before fitting; package,
  example and feature-cache hashes were checked against the committed config.
- No audio stream is present in the HABIT train/validation media. This study
  does not cover four-modality quality.

## OOF learning curves

Sample-weighted rows show Accuracy / NLL / Brier / ECE. Macro task-group
Accuracy / NLL are included in parentheses. These are grouped OOF training
diagnostics, not development-validation results.

| Arm | Epoch 1 | Epoch 2 | Epoch 4 | Epoch 8 |
|---|---:|---:|---:|---:|
| Text | 44.6% / 1.023 / .615 / .074 (48.8% / .965) | 42.6% / 1.038 / .623 / .056 (47.5% / .968) | 42.6% / 1.060 / .633 / .093 (48.8% / .979) | 41.6% / 1.123 / .654 / .122 (48.7% / 1.014) |
| Video | 45.5% / 1.022 / .614 / .087 (51.5% / .965) | 43.6% / 1.033 / .620 / .055 (48.4% / .966) | 43.6% / 1.051 / .628 / .094 (49.2% / .975) | 43.4% / 1.159 / .664 / .138 (51.6% / 1.035) |
| State | 46.5% / 1.022 / .615 / .077 (52.8% / .966) | 43.6% / 1.034 / .620 / .050 (48.4% / .967) | 44.6% / 1.050 / .628 / .077 (48.8% / .980) | 42.7% / 1.078 / .632 / .101 (48.6% / .984) |
| Video + state | 45.5% / 1.023 / .614 / .084 (51.5% / .965) | 42.6% / 1.035 / .621 / .068 (47.5% / .966) | 43.6% / 1.055 / .629 / .103 (49.2% / .976) | 38.1% / 1.141 / .671 / .164 (45.5% / 1.018) |

Training CE (mean of fold mean-batch CE) at epochs 1/2/4/8 was:

| Arm | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Text | 1.013 | 0.956 | 0.922 | 0.879 |
| Video | 1.013 | 0.959 | 0.925 | 0.802 |
| State | 1.014 | 0.959 | 0.917 | 0.819 |
| Video + state | 1.013 | 0.959 | 0.921 | 0.810 |

The prior 202-row grouped OOF is useful context but not a paired causal
comparison: the expanded train set changes the fold membership and includes
the old rows. At epoch 2 the earlier video/state OOF Accuracy was about 50.5%
and fusion 49.0%; the expanded run was 43.6%, 43.6% and 42.6%, respectively.
This does not support a benefit from the added rows under this fixed readout
protocol, but it is not an independent replication.

## Runtime and files

- CPU training: 77.54 seconds, 4 threads; text feature extraction: 4.32
  seconds; 2,560 optimizer updates. CUDA was not used; cloud cost was zero.
- Frozen package path:
  `C:\CodexLocalArtifacts\pretrained-reuse-r2\habit-data-scaleup-seed17-attempt-01\decision-package`.
- External OOF output path:
  `C:\CodexLocalArtifacts\pretrained-reuse-r2\habit-data-scaleup-seed17-attempt-01\train-only-grouped-oof`.
- Committed config:
  `configs/pretrained_reuse/habit_train_only_grouped_oof_data_scale.yaml`;
  SHA-256 `ba26367e4aa5a36b77bc007d1062e7d705f8efe2b6a136131ebe330d382718c8`.
- Package report SHA-256:
  `062013df3edf0c63c84e4c43edeb454430cf896a0ed75d5500011d2f492d505e`.
- Transition examples SHA-256:
  `87b405ba343f4a251bd3450f306514e552a66b860b9cc558fd65880c09d2f8f5`.
- Feature cache SHA-256:
  `5c494417aa0946041881f5a9305eb852d617769dc3261dda99c9772bb6301928`.
- OOF report SHA-256:
  `6b5032ce706ecae94b41374bc88f4431b8b5c6ca90d72aa8ad95e1dbdc75b947`.
- Paired component-bootstrap SHA-256:
  `716ea708e960d3e94632a5384fe53827ad32369ea99c3c0940c96ecbe964d86c`.

The OOF report, folds, predictions and bootstrap output remain outside Git.
No original Teacher/model weights, media, package or checkpoints were changed.
