# Video Teacher v2 Design

Date: 2026-10-04

## Purpose

Upgrade the video side of Tiny Omni Decision without invalidating the currently running Teacher v1 experiment.

Teacher v1 remains a frozen baseline. Its current run must finish with the exact existing corpus, config, frame count, LoRA target set, and validation protocol. No in-place edits to the active v1 artifacts are allowed.

The new video work targets two suspected bottlenecks:

1. temporal observation bandwidth is too small because the current processor path forces four video frames;
2. adaptation capacity is too narrow because LoRA is applied only to decoder `q_proj` and `v_proj` at rank 16.

A separate dataset issue is also addressed: current CLEVRER normalization keeps only `descriptive` questions, while upstream CLEVRER also contains explanatory, predictive, and counterfactual video reasoning tasks.

## Product Goal

Preserve the four-modality Decision interface while making Video genuinely temporal.

The target remains calibrated option probabilities for text, image, audio, and video. Video should require temporal evidence where appropriate rather than acting as an image-classification proxy.

Success is measured by:

- validation improvement without regressing already-strong modalities;
- higher Video Accuracy and lower Video NLL/Brier on clean held-out data;
- no train/validation media leakage;
- no tuning against the sealed audit;
- reproducible A/B comparisons against Teacher v1;
- no unnecessary full-model fine-tuning.

## Non-Goals

- Do not modify or restart the current Teacher v1 run.
- Do not unfreeze the full base model.
- Do not unfreeze modality encoders in the first Video v2 experiments.
- Do not change the final typed-option Decision API.
- Do not combine ternary conversion, Recovery, pooling, or lightweight readout work into this experiment.
- Do not use the sealed audit for hyperparameter selection.

## Current Baseline

The active Teacher v1 path currently uses:

- `video_num_frames = 4` hard-coded in the processor path;
- decoder LoRA on `q_proj` and `v_proj` only;
- LoRA rank 16, alpha 32, dropout 0.05;
- frozen modality encoders and projector;
- fixed learning rate `5e-5`;
- joint text/image/audio/video training;
- CLEVRER `descriptive` questions only;
- validation every 128 optimizer steps;
- validation-only checkpoint selection.

This baseline is preserved unchanged for comparison.

## Design Principle

Change one major causal factor at a time.

The first experiments separate:

1. frame-count limitation;
2. LoRA target coverage;
3. LoRA rank;
4. dataset task coverage.

This avoids attributing an improvement to the wrong cause.

## Configuration Changes

Add explicit training configuration fields instead of hard-coding runtime behavior.

### Video sampling

Add:

```yaml
training:
  video_num_frames: 4
```

Teacher v1 config remains explicitly at 4.

Video v2 candidates use 8 frames first. A 12-frame candidate is permitted only after the 8-frame run is measured.

The processor must read the configured value rather than forcing `num_frames: 4`.

The existing maximum sequence-length guard remains authoritative. A run must fail closed if the processor output exceeds the configured sequence limit.

### LoRA target policy

Add an explicit target policy:

```yaml
training:
  lora_target_policy: qv
```

Supported policies:

- `qv`: decoder `q_proj`, `v_proj`;
- `attention`: decoder `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- `decoder_all_linear`: decoder attention projections plus `gate_proj`, `up_proj`, `down_proj`.

All target discovery remains restricted to `model.language_model.*`. A generic model-wide `all-linear` selector must not accidentally adapt vision/audio/projector modules.

### Rank and scaling

Start Video v2 target-coverage experiments at rank 16.

Only after `decoder_all_linear` rank 16 is measured may rank 32 be tested.

If rank 32 is implemented, expose rsLoRA as a config switch and use rsLoRA for that candidate rather than changing several other variables at the same time.

### Learning-rate schedule

Keep the current Teacher v1 schedule unchanged.

For new broad-target Video v2 candidates, add configurable scheduler support:

```yaml
training:
  lr_scheduler: cosine
  warmup_ratio: 0.03
```

The first broad-target candidate uses `5e-5`, 3% warmup, cosine decay. If instability appears, lower learning rate in a separate candidate; do not silently alter the running experiment.

## Video Dataset Design

### Preserve temporal descriptive questions

Do not discard all descriptive CLEVRER questions. Many descriptive questions are temporal, for example questions involving:

- first/last collision;
- objects entering or exiting;
- events before or after another event;
- state at the beginning or end of the video.

These remain useful grounding tasks.

### Add native video reasoning types

Extend CLEVRER normalization to support:

- explanatory;
- predictive;
- counterfactual.

Do not force these into the existing open-ended taxonomy format.

For CLEVRER multiple-choice reasoning records, normalize each upstream choice as a binary Decision example:

```text
state: synthetic CLEVRER video
question: <original question + candidate statement>
options: [wrong, correct]
target: upstream choice label
media: same underlying video
```

All derived choices from one CLEVRER question and all questions from one video remain in the same split.

This preserves the existing `DecisionExample` single-target interface and avoids adding a new multilabel loss path.

### Video training mix

The first video-native corpus targets approximately:

- 20% temporal descriptive;
- 30% explanatory;
- 30% predictive;
- 20% counterfactual.

These are sampling weights, not requirements to duplicate scarce examples. Prefer unique questions and unique videos over repetition.

If upstream availability prevents the exact proportions, record the realized counts and use the closest deterministic mix without synthetic duplication.

### Static descriptive filtering

Purely static descriptive questions may remain in the source corpus but should not dominate the video training share.

A question should count as temporal descriptive when its text or upstream program semantics references motion, collision, entry/exit, beginning/end state, or temporal ordering. The implementation must use a deterministic classifier based on known CLEVRER subtype/program metadata where available, with a documented text-pattern fallback for source records that lack that metadata.

## Split Integrity

The current leakage protection remains mandatory.

Group by underlying CLEVRER video identity.

No video may appear across train, validation, or sealed audit.

Derived binary choice examples inherit the same source-asset identity as the parent video.

Cross-split checks continue to reject shared source IDs, source assets, media identities, and normalized content.

## Experiment Ladder

Run candidates in this order and preserve every artifact.

### Baseline V1

- 4 frames
- q/v LoRA
- rank 16
- existing data

Purpose: immutable reference.

### Candidate A: frame bandwidth

- 8 frames
- q/v LoRA
- rank 16
- same v1 data

Purpose: isolate whether four frames are the main Video bottleneck.

### Candidate B: target coverage

- 8 frames
- decoder-all-linear LoRA
- rank 16
- same v1 data

Purpose: isolate whether q/v-only adaptation is the main capacity bottleneck.

### Candidate C: rank capacity

Only if B still plateaus.

- 8 frames
- decoder-all-linear LoRA
- rank 32
- rsLoRA enabled
- same v1 data

Purpose: test low-rank capacity after target coverage has already been broadened.

### Candidate D: temporal resolution

Only if A/B/C indicate continuing benefit from additional frame coverage and memory permits it.

- 12 frames
- best adapter policy from A/B/C
- sequence limit increased only as required by measured processor output

Purpose: test whether more temporal observation remains useful.

### Candidate E: video-native task mix

Use the best structural candidate from A-D and train with the expanded CLEVRER temporal/causal corpus.

Purpose: measure the effect of task coverage separately from frame count and LoRA capacity.

## Selection Rules

Use the existing validation-only selection process.

Do not select a checkpoint or config from the sealed audit.

Track at minimum:

- per-modality Accuracy;
- per-modality NLL;
- per-modality Brier;
- per-modality ECE;
- macro Accuracy;
- minimum-modality Accuracy;
- train/validation gap;
- unique examples and unique underlying assets;
- realized sample mix;
- peak VRAM;
- wall time.

Do not accept a Video gain that materially damages Audio, Image, or Text without an explicit later trade-off decision.

## Overfitting Diagnostics

Do not interpret a rising Video validation score alone as proof of general video understanding.

For Video v2, additionally report metrics by CLEVRER reasoning category:

- temporal descriptive;
- explanatory;
- predictive;
- counterfactual.

Also report per-video grouping counts and accuracy by question type.

If train accuracy rises while category-level validation stalls, treat it as capacity/data mismatch or overfit rather than continuing indefinitely.

## Files Expected to Change During Implementation

Likely modifications:

- `src/tiny_omni_decision/training.py` — config fields, LoRA target policy, video frame count;
- `src/tiny_omni_decision/trainer.py` — scheduler wiring and LoRA options;
- `src/tiny_omni_decision/dataset.py` — CLEVRER native reasoning normalization;
- `configs/decision/` — Video v2 experiment configs;
- `manifests/` — new Video v2 corpus catalogs if required;
- `tests/test_training.py` — frame/target/scheduler configuration tests;
- `tests/test_dataset.py` — CLEVRER explanatory/predictive/counterfactual normalization and split tests;
- `docs/TEACHER_V1.md` — preserve v1 as baseline and point to v2;
- a new Video v2 experiment document for measured results.

Exact implementation file changes are defined in the implementation plan after this design is approved.

## Worker Handoff

The worker must:

1. let the current Teacher v1 run finish unchanged;
2. preserve its best checkpoint, history, hashes, and metrics;
3. not regenerate or overwrite the frozen v1 corpus;
4. implement Video v2 on a separate branch after this design is approved;
5. run the experiment ladder one causal change at a time;
6. keep all candidate artifacts instead of overwriting them;
7. never use sealed-audit results for iterative tuning.

The current run is useful even if it plateaus because it is the reference required to measure whether frames, LoRA coverage, rank, or task mix actually caused the improvement.
