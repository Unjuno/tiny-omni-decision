# Bounded frozen joint-feature readout comparison

Completed: [measured results and learning curves](R2_JOINT_VIDEO_READOUT_RESULTS_20261013.md).
Both fixed50-update runs finished; this readout was not adopted.

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

## Validation input admission correction

First validation extraction stopped after six complete videos on the literal
candidate "Under the fourth object from the left." The train-derived whitelist
had only first/second/third. Added fourth with the identical semantic template;
no training input changes, no rows/options removed or relabeled. All 1,542
validation candidate statements now pass, each30–34 tokens, no truncation.
Preserve partial all-in-one-tiny-valid-features-01. Restart full extraction into
all-in-one-tiny-valid-features-02; train-features-01 continues unchanged.
Regression tests include literal fourth-ordinal preservation.

## Readout admission tests

On the real one-video CPU feature smoke, replacing the two-logit ITM layer with
its positive-minus-negative weight vector reproduces option probabilities to
max absolute error5.960464477539063e-08 (admission tolerance1e-6). This is192
trainable parameters; the common scalar bias cancels in option softmax.
Feature admission now rejects partial runs, unexpected protocol settings and
frame IDs that differ across question conditions or black/normal controls.
Synthetic readout tests verify the 51 recorded points (epoch0 plus50 updates),
minimum-validation-NLL selection, separate final weight and exact best reload.
Actual corpus training remains dependent on both full feature extractions.

## Completed training-feature cache admission

Train extraction completed99 videos /198 questions in600.539seconds CPU,
396 rows covering normal and black inputs. Each mode has shape[198,3,192].
The full prior diagnostic's396 probability/identity/frame records match exactly
when removing the newly added feature vectors; no extraction behavior drift.
Cache SHA2568d061543600446f0dfd659feeb1e88c6c039c7f8cfab3104141c92a5fe9be3ef.
Admission saved under all-in-one-tiny-train-features-01/admission.json.
Validation extraction remains in progress; no actual readout fit yet.
