# HABIT temporal Decision readout protocol — 2026-11-21

## Status before fitting

`CPU_READOUT_SMOKE_PASS; QUALITY_NOT_MEASURED`. A two-example smoke passed
for the text-only, video-only, state-only and paired readouts. Each returned
finite masked logits in supplied option order. It used local frozen features
and MiniLM; it did not update parameters. The smoke receipt is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\habit-temporal-decision-smoke-20261121-attempt-04\smoke-report.json`
(SHA-256 `e05525305f5189d43984d34a4f2db123f361d48c27975417fa0765a7229d147d`).

## Frozen task and feature inputs

This pilot asks: given a high-level instruction, causal front-view video, and
robot0 state, which supplied task substep should happen next? The exact target
is the next `low_level_task_index` mapped to the same ordered, deduplicated
candidate action text used by the dataset adapter. No free-form action
generation is attempted.

Input package:
`C:\CodexArtifacts\pretrained-reuse-r2\habit-transition-decision-data-20261121-attempt-01`.
It contains 202 train and 76 development-validation transitions, with 36 and
6 task groups, respectively. The group and media identity splits are frozen
and disjoint. All examples use eight pre-onset VideoPrism-B frame features and
eight time-aligned 14-D robot state rows. The fixed package and source hashes
are in [the feature pilot report](R2M_HABIT_TRANSITION_FEATURES_20261121.md).

Frozen encoders:

- VideoPrism-B: 114,365,184 parameters; revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`; checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- MiniLM-L3: 17,389,824 parameters; revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Both are frozen. Text embeddings are masked means of the task instruction
  and each action candidate. Candidate features are `[question, option,
  |question-option|, question*option]`.

## Fixed comparison

One CPU seed-17 run fits four arms on the same 202 examples:

1. Text/options only: a 196,865-parameter candidate MLP.
2. Video + text/options: the existing causal temporal Decision readout, with
   state explicitly absent.
3. Robot state + text/options: the same readout, with video explicitly absent.
4. Video + state + text/options: the same readout with both streams present.

Each multimodal readout has 370,819 trainable parameters. Counting the two
frozen encoders, the paths contain 131,951,873 parameters for text-only and
132,125,827 for each temporal arm before quantization. These are parameter
counts, not serialized package-size measurements.

The fixed schedule is 80 epochs, batch size 32, AdamW learning rate 0.001,
weight decay 0.0001, cross-entropy loss, and candidate-order randomization on
every training batch. Every arm processes all 202 unique train examples once
per epoch (560 updates and 16,160 presentations per arm). Early stopping is
disabled to observe the complete learning curve and allow late improvement.
Best checkpoints are selected by minimum validation macro task-group NLL with
`min_delta=0.0001`; last checkpoints are kept separately. Sensor channel means
and standard deviations are fit on train rows only.

Report sample-weighted and macro task-group Accuracy, NLL, Brier and 15-bin
ECE, individual task-group metrics, predictions and probability vectors. The
paired checkpoint also gets fixed validation interventions: video missing,
state missing, cross-group video shuffle, and an independent cross-group state
shuffle. These post-selection ablations are input-dependence diagnostics, not
new blind tests. Paired differences against the other arms use a seed-17,
2,000-draw task-group bootstrap. With only six held-out groups, its intervals
are coarse and selection-affected.

## Safety and interpretation bounds

The run is CPU-only. It does not load sealed-audit/test data, use action
columns, change the source split, or quantize any weights. No product promotion
is permitted from these validation results. The pilot is video + robot state +
text; it does not cover audio or establish the overall four-modality 95–98%
objective. Six validation groups are too few for broad generalization claims.
Any result will be limited to plumbing and this small development pilot unless
a larger rights-cleared grouped validation cohort is admitted.

The effective settings are fixed in
[`habit_temporal_decision.yaml`](../../configs/pretrained_reuse/habit_temporal_decision.yaml).
