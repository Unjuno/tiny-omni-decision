# Physion++ train-only scenario holdout protocol — 2026-11-28

## Question

The existing 652-row Path A train set contains ten simulation scenarios. Its
148-row development validation contains all ten and is used for checkpoint
selection, so it does not measure transfer to a new scenario. This fixed
diagnostic asks whether the frozen-feature readout transfers across scenarios
within the already-used training corpus. It is data-generalization evidence,
not a new blind evaluation.

## Frozen design

- Input data: existing train JSONL only, SHA-256
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`.
- Frozen feature cache: pinned VideoPrism-B 8-frame/288px train cache at
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-videoprism-features-20261128-attempt-02`.
- Split: leave one entire `scenario` out per fold; verify each row appears in
  exactly one held-out fold and no scenario+scene-seed group crosses a fold.
- Architecture and optimizer: identical 770-input, 128-hidden shared candidate
  scorer; 98,817 trainable parameters; AdamW, LR `0.001`, weight decay
  `0.001`, batch 32, per-question option-order randomization.
- Training: seed 17 independently reset in each of ten folds; fixed 200 epochs;
  no early stopping, fold-model selection, or held-fold feedback during fitting.
  Evaluate the final fixed-budget model once on its held-out scenario.
- Controls: each fold's training-label prior, the same fold model with video
  features zeroed, and a deterministic shuffled-video diagnostic.
- Report: each held-out scenario's count/label balance and metrics; pooled
  out-of-fold Accuracy/NLL/Brier/ECE; equal-weight macro Accuracy and minimum
  scenario Accuracy; group-paired bootstrap versus zero-video control using
  10,000 replicates and PCG64 seed `20261128`.
- Resources: CPU only. No GPU, cloud, publisher test or sealed audit data.

## Interpretation limits

All records have already been used by the previously fitted full-train Path A
candidate. The cross-validation models hold their own scenarios out, but this
analysis is post hoc on known development/training data and does not provide an
independent product-quality result. Ten scenarios are a small cluster count;
report individual scenario results and avoid broad generalization. Do not
change frame count, architecture, epochs or seed after observing this run. A
negative or mixed result is retained as-is. No BERT, shared-Lens, or weight
merging experiment is part of this protocol.

## Execution record

This file is committed before the experiment. The measured report, per-row
predictions, fold checkpoints and loss curves will be placed under a fresh
`C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-path-a-scenario-holdout-20261128-attempt-01`
directory and summarized here only after execution completes.

## Preflight result (before fitting)

The frozen train manifest produced 10 folds over all 652 records and 492
scenario+scene-seed groups. Every record is held out exactly once, and no
group crosses the train/held-out boundary. No development validation or
publisher test data was loaded for this preflight.

| Held-out scenario | Train rows | Held-out rows | Held-out labels No / Yes | Held-out groups |
|---|---:|---:|---:|---:|
| `bouncy_platform_pp` | 565 | 87 | 55 / 32 | 43 |
| `bouncy_wall_pp` | 563 | 89 | 43 / 46 | 53 |
| `deform_clothhang_pp` | 568 | 84 | 45 / 39 | 84 |
| `deform_clothhit_pp` | 568 | 84 | 32 / 52 | 57 |
| `friction_cloth_pp` | 596 | 56 | 28 / 28 | 55 |
| `friction_collision_pp` | 602 | 50 | 28 / 22 | 35 |
| `friction_platform_pp` | 601 | 51 | 34 / 17 | 46 |
| `mass_collision_pp` | 599 | 53 | 21 / 32 | 36 |
| `mass_dominoes_pp` | 606 | 46 | 29 / 17 | 40 |
| `mass_waterpush_pp` | 600 | 52 | 24 / 28 | 43 |

Fitting status at protocol commit: **not started**.
