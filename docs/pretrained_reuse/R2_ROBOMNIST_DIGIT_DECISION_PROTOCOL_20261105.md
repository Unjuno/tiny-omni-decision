# RoboMNIST task-native digit Decision protocol — 2026-11-05

## Fixed task

Given one original training repetition, choose which of ten supplied digits the
robot drew. The upstream `Ac` label (0–9) is target supervision only; robot,
speed, repetition and filename metadata never enter model features. The runtime
control outputs a ten-class probability vector in the caller's supplied option
order. This is a narrow synthetic controlled-lab task, not natural handwriting,
robot control or broad world reasoning.

## Split and leakage controls

- Frozen manifest and feature cache from the earlier audio/video correspondence
  development are inputs; new model fits are restricted to its 780 TRAIN media
  groups. Original validation arrays and sealed audit are never loaded.
- Group all repetitions by full `Ar x velocity x Ac` condition into deterministic
  five folds (seed17). This tests transfer across held-out task conditions and
  prevents repetitions of one action condition crossing folds. All source groups
  remain intact. Require all 60 conditions and all ten digit labels before fit.
- There are only six condition groups per digit (2 arms x 3 speeds), so held-out
  uncertainty by condition is limited. The same source was used in earlier
  metadata development; these are development folds, not blind evaluation.
- Fit mean/std pooling normalization inside each fold using only fit-fold source
  groups. No held condition contributes to normalization or training.

## Compared models

One affine ten-choice readout each over frozen pooled features: audio only
(7,690 parameters), video only (15,370), and concatenated audio+video (23,050).
All use seed17, AdamW, lr0.01, weight decay0.0001, gradient clip5, full-fold
batch and 400 updates. Record steps1/25/50/100/200/400; no step is selected for
promotion. Report train/OOF Accuracy, NLL, multiclass Brier, ECE-15, mean
confidence, per-class counts/Accuracy, predictions and save/reload parity.
Encoders are frozen. Audio windows and video frame tokens were sampled on no
shared clock; clip-level pooling does not assert temporal alignment.

Joint vs unimodal measures modality complementarity on this task but remains
vulnerable to correlations among the 60 synthetic conditions. A joint gain is
not sufficient to claim a product-grade both-required task. No sealed-audit or
model promotion follows this internal diagnostic.
