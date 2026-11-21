# TallyQA-VG TRAIN-only grouped cross-validation protocol

## Purpose and scope

This fixed experiment estimates how the existing frozen VideoPrism-B plus
MiniLM readout transfers to held-out connected groups within the source TRAIN
partition for the project-created four-choice TallyQA adaptation. It follows
the ROADMAP instruction to diagnose the weak natural-image task using grouped
TRAIN-only evidence before changing the shared architecture.

The result is retrospective development evidence. The prior readout and task
settings were selected using an already exposed development validation set,
and this is not an independent product-quality estimate. The original TallyQA
task is open-response; the four-choice conversion is not directly comparable
to its official metric. This experiment does not reopen or use the previous
validation predictions, official test, or sealed audit.

## Frozen inputs

- Source/task: pinned TallyQA dataset at
  `46cdc649ec79c3dcc2720ff227ad07d7ee51da6f`; use the already prepared AMT
  Visual Genome TRAIN rows only. Existing source admission is limited to this
  local development adaptation; this protocol does not promote it to a product
  benchmark.
- Original task manifest SHA-256:
  `59c1c03edf17a8da6c870d4827d033af58759c8300f953b60cb0abbff98f8a45`.
- Frozen VideoPrism-B revision:
  `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`,
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Frozen MiniLM-L3 revision:
  `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`,
  weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Frozen TRAIN image-feature cache SHA-256:
  `ddd738a362d7b31762c676dc178fed0183ddce8052ffaac4bbc93f57ef2753a6`.
- Existing readout config SHA-256:
  `60857ddebb2ab0d741d46f9e08b077160069d536c80dbd862654ef5b1c481164`.
- External run root (must not exist before preflight):
  `C:\CodexArtifacts\pretrained-reuse-r2\tallyqa-vg-train-groupcv-v1-attempt-03`.

The input-builder writes a separate manifest containing only the source TRAIN
rows and the SHA-256 of its parent task manifest. The model runner requires
this `split_scope=train_only` manifest and rejects any validation/test/audit
fields. The feature loader reads only `train_ids`, `train_media_sha256`, and
`train_features` from the existing NPZ. It does not access its validation
feature members. Text features are encoded only for TRAIN questions and their
four fixed count candidates.

## Fixed evaluation

- Seed 17; five outer folds over transitive connected components joining
  shared `image_id` or NFKC/casefold/alphanumeric-normalized question text.
- Deterministic, label-blind greedy assignment sorts components by descending
  row count and stable hash, then assigns each to the least-loaded fold. Every
  row from a component stays in one outer fold; answer-position support is
  reported but does not influence assignment.
- For each outer fit, assign five folds over the remaining groups and use
  inner fold 0 (about 20%) only for checkpoint selection. The outer fold is
  scored once after the earliest minimum inner-stop NLL checkpoint is loaded.
- Compare two fixed arms: frozen image + question + candidate features, and
  question + candidate features only. Keep the existing MLP architecture,
  16 epochs, batch size 64, AdamW learning rate 0.001 and weight decay 0.0001.
  Use the same fold IDs and seed for both arms.
- Apply a post-fit image-zero control to the joint checkpoint. It cannot
  select a checkpoint or alter the outer OOF fit.
- Report pooled OOF Accuracy, NLL, multiclass Brier and ECE, fold-wise metrics,
  and paired differences between joint and question-only. Use a 5,000-draw
  bootstrap over the same transitive image/question components (seed 17); all
  questions from every image sharing a component move together.
- Save OOF predictions in original TRAIN question order, selected readout
  states and per-fold histories outside Git. Record hashes and effective split
  assignments in the preflight receipt before fitting.

## Interpretation rule

This is one fixed grouped-CV measurement, not a hyperparameter search. Do not
change the task conversion, folds, architecture, seed, epochs, thresholds or
sampling after seeing results. Image input use is supported only if zeroing
image features harms OOF proper scores as well as accuracy. A joint improvement
over question-only must be assessed with the paired image-cluster interval and
all proper scores; mixed or uncertain results do not justify more tuning on
this source. Regardless of outcome, do not claim general image or Omni quality,
R2-M closure, product admission, or success on the original open-response
benchmark. Keep official validation, official test and sealed audit closed.

## Run status

Protocol frozen before fitting. Preflight attempts 01 and 02 stopped before
model/text-feature inference: attempt 01 rejected repeated questions per image
as duplicate image rows; attempt 02's label-stratified greedy split produced
two outer folds with only 18 rows each. Both attempts are preserved outside
Git; neither fitted or scored a model. The fixed label-blind component-size
assignment and attempt 03 receipt were frozen before fitting. The source
commit, code and input hashes, environment, runtime, fold-level results, and
decision are recorded in [the results report](R2_TALLYQA_VG_TRAIN_GROUP_CV_RESULTS_20261121.md).
The runner-reported elapsed time covers the fold-fit/scoring/bootstrap section
only; it excludes initial input and feature preparation. Exact command history
was not persisted, and peak VRAM was not recorded. Selected checkpoints and
OOF predictions were hash-verified and replayed exactly. Official validation,
official test, and sealed audit remain unopened. Do not retune this source or
claim that image input improves held-group performance based on this result.
