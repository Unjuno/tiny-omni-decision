# Bounded frozen joint-feature readout comparison

Purpose: distinguish a weak pretrained ITM classifier from unusable frozen
joint video/text representations. Existing modular trajectory/scene readouts
do not test these jointly contextualized representations.

Keep All-in-one matching backbone and pooler frozen, 192-dimensional cls_feats,
same pinned Perception occlusion pairs, three frames, compact statements,
CPU FP32 and seed17. Preserve all earlier results. Checkpoint licensing remains
REVIEW: research-only experiment, no product adoption or weight redistribution.

## Fixed comparison

- Extract normal and black-image features for all 99 train / 257 development
  validation videos, preserving both conditions and all three options.
- Two separate candidate scorers: normal features and black-image features,
  identical linear192-to-1, no bias (shared option bias cancels under softmax).
  Exactly192 trainable parameters per scorer. No encoder adaptation.
- Initialize each scorer to the pretrained ITM positive-minus-negative weight,
  preserving its option probabilities. Record initial performance.
- Seed17, full-batch AdamW, learning rate0.001, weight_decay0.01, 50 epochs,
  candidate cross-entropy; no data augmentation beyond the pinned feature cache.
- Choose best validation NLL independently, strict improvement, earliest ties.
  Record every epoch, retain best and final, verify exact save/reload predictions.
- Report Accuracy/NLL/Brier/ECE, beginning/end, both-correct and changed-target
  pairs, train/validation gap, runtime, artifact bytes and resident parameters.
- Report paired video-cluster percentile95 intervals, 10,000 draws, seed17.
  No extra seeds, prompt search, calibration or validation-driven capacity sweep.

Do not adopt on aggregate Accuracy alone. Evidence of useful video requires
improvement over the same-capacity black-image scorer, including changed-target
pair behavior. A failed comparison closes this readout route without claiming a
model-family limit. These are previously observed development videos, not a blind
test; sealed audit remains untouched.

## Preflight evidence

GPU feature smoke failed before predictions: upstream visual_embed indexes a CPU
patch_index with CUDA select tensors. No upstream source edit was made. Preserve
the failed artifact directory all-in-one-tiny-feature-smoke-01.
CPU smoke passed: four rows x three candidates x192 finite features; probabilities
exactly match the earlier normal/black diagnostic for the same video.
Artifact: C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-feature-smoke-cpu-01.
CPU path avoids mixing a GPU compatibility patch into this readout experiment.
Full training-feature extraction is the next step; no completed training claimed.
