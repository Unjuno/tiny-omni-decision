# DROID: executed train-only nested temporal representation comparison

## Result and decision

On the 275 frozen DROID training recording groups, an invertible temporal
representation gave a small improvement over raw sensor flattening under
predeclared nested cross-validation. Pooled outer macro NLL fell from 2.2347
to 2.1833 and Accuracy rose from 40.36% to 41.82% (four additional correct
episodes). Outer macro NLL improved in three folds and worsened in two.
Neither arm correctly classified any `stop`, `x-`, `y-` or `y+` outer examples.
This does not establish a useful general decision model, joint-modal benefit,
or the 95–98% quality target.

Retain the result without extending the hyperparameter grid or opening
validation. This changes a weak regularization/representation result into an
executed comparison, not a capacity diagnosis. The next product-facing
experiment still needs a prospectively defined decision task with evidence
that multiple inputs are required. Do not use this outcome to declare xLSTM,
LSTM, longer training or pretrained video representations incapable of helping.

## Frozen design and isolation

The [plan/protocol](../superpowers/plans/2026-10-27-droid-temporal-train-cv.md)
was written before execution; its hash is retained in `protocol.json` and
the result receipt. User delegated experimental strategy. This comparison
follows the [train persistence diagnostic](R2_DROID_TRAIN_TEMPORAL_20261027.md).

Two representations use exactly the same original eight 14-dimensional sensor
states and seven-way targets:

- `raw`: flatten all eight states to 112 values.
- `delta`: concatenate the last 14-value state and seven successive 14-value
  coordinate increments, also 112 values. This is invertible; no information,
  time point, commanded-action history or encoder output is added. Raw angular
  coordinate increments are not angle-unwrapped or interpreted as physical
  velocity. Varying temporal index gaps are not normalized.

Each arm uses the original L2 multinomial linear head, 791 parameters. A
different invertible transform changes standardization, the L2 penalty and
possible clipping; it does not expand the underlying unconstrained linear
hypothesis class. Parameter equality alone is not equality of inductive bias.

Five seed-17 outer folds are stratified by target at recording-group level.
Within each outer training set, four grouped folds select L2 among exactly
`100, 1000, 10000` using macro NLL, with ties favoring larger L2. Standardization
uses each actual fit's training rows only. Both arms share identical group
partitions. Each selected head then fits its entire outer training set and
predicts only its held outer fold. Every recording gets one outer prediction
per arm. All 120 inner fits plus ten selected outer fits converged within the
fixed 2,000-iteration cap; no retry or search extension occurred.

This is a development diagnostic on an already inspected training cohort,
not a new blind evaluation. Inner CV chooses regularization; outer predictions
estimate the fixed procedure within this cohort. The five folds overlap in
training data and are not five independent experiments. No inferential
confidence interval or significance claim is made. Recording groups do not
establish independent operators, sites or robot sessions.

## Measured outer predictions, 275 train recordings

| Rule / representation | Accuracy | Macro Accuracy | NLL | Macro NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| Outer-training class prior | 37.09% | 14.29% | 1.6576 | 2.2584 | 0.7651 | 0.0002 |
| Raw sensor states | 40.36% | 18.05% | 1.6425 | 2.2347 | 0.7568 | 0.0643 |
| Final state + coordinate increments | 41.82% | 18.55% | 1.5712 | 2.1833 | 0.7239 | 0.0443 |

The prior uses class frequencies only from each outer training set. Its small
ECE does not imply useful discrimination. Metrics use the exact original
seven-class implementation, including ten confidence bins and unnormalized
multiclass Brier sum. All seven classes are retained. The receipt includes
per-class counts, Accuracy and NLL, all inner candidate scores and exact folds.

| Outer fold | Held recordings | Raw L2 | Delta L2 | Raw Accuracy | Delta Accuracy | Raw macro NLL | Delta macro NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 58 | 100 | 1000 | 34.48% | 39.66% | 2.2952 | 2.1885 |
| 1 | 57 | 100 | 100 | 42.11% | 43.86% | 2.1575 | 2.1239 |
| 2 | 56 | 100 | 100 | 42.86% | 39.29% | 2.2166 | 2.2220 |
| 3 | 53 | 100 | 100 | 41.51% | 39.62% | 2.3023 | 2.3243 |
| 4 | 51 | 1000 | 100 | 41.18% | 47.06% | 2.2102 | 2.0922 |

The previous 94.55% command-persistence result was a deterministic train
diagnostic using extra command history; it is not a matched baseline here.
The earlier DROID development-validation scores are not mixed into this table.
No development validation rows or predictions were scored in this experiment.

## Actual artifacts and reproducibility

Source worktree base: `856116ff655e6cf08abb83eb5276f928103d2e4a`.
Pinned original readout/extractor:
`eba818067c6a9ef9089288205c26f6a8254ca6e5`. New code was uncommitted during
execution and is bound by its script/protocol hashes. The pinned extractor's
`read_rows` establishes train-first order. All train labels match the immutable
train JSONL; summary, sensor and label file hashes match original receipts.

New output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-train-temporal-cv-20261027-attempt-01/`.
Contains protocol, pinned readout/extractor source, all outer train probabilities,
ten fold-specific NPZ heads (101,350 bytes total), and complete result JSON.
No final all-train/product head was created. Weights remain outside Git.

The sensor/label containers also contain development validation rows. Their
whole bytes were hashed, but only the first 275 train rows were copied through
read-only memory maps. No task-manifest validation examples, validation feature
rows or sealed audit were loaded for fitting/evaluation. The extraction summary
contains aggregate validation counts. All input file hashes matched again after
execution. Existing data, features, model weights and run outputs are unchanged.

An independent saved-head replay checked all ten outer fits: means and scales
exactly equal their outer training features, inner partitions exclude the
outer-held rows and partition all outer training rows, every head contains
791 trainable values, and saved probabilities reproduce within `1e-12`.
Unit tests additionally check invertibility, fixed width, finite/shape rejection
and outer prediction rows excluded from fitting and hyperparameter selection.

Execution: October 26, 2026 16:15:29–16:15:32 UTC (October 27 JST).
Nested fitting/result generation took 3.21 CPU seconds, excluding startup and
input preflight. Environment: existing Python 3.11.16, NumPy 1.26.4, SciPy
1.17.1, Windows; `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`. No upgrades,
GPU, encoders, download or cloud cost. Peak RAM was not measured.

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
& C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe `
  -m scripts.compare_droid_train_temporal `
  --train C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-prep-attempt-02/train.jsonl `
  --feature-cache C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-features-attempt-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/droid-train-temporal-cv-NEW
```

Hashes:

- Train: `4e14850bbbe2eee2a96116a97ca935dfd5dea91d197c00639ea7ed02e2d93341`.
- Sensor: `de8dc6de9afec85932b31d08d94ed58d097896e03ad1098813fd9f2c7ce372b9`.
- Labels: `372579d3a5b639aa0170684abc9863b24d10ab3dee6e3821a3a261817fab80b1`.
- Executed script: `6e4fd7760592c29b0bf2601db248286bf071cbf9554b7f6ee2f7bd771af1c38f`.
- Original readout: `186fad84f223748bd4670903c7b85e61aa6513eed8e311065bf58d17176d7728`.
- Protocol plan: `397d9dff5f982009f1bf68a68871182feb2271135a7078b5859042bae1ac587f`.
- Outer predictions: `f020b2e3bcd0eb36ccc4117b2946a152cd98b5db27b9efc0e954a21214d7d6aa`.
- Result receipt: `8b44d08afe358f5df2c37bca93b2a14998ede472acc8d031c2d2e1d11eb12867`.

See the [full result receipt](R2_DROID_TRAIN_TEMPORAL_CV_20261027.json) for
remaining input/source hashes, model hashes and all attempted scores.
Ruff passed; CPU suite **2,630 passed, 11 skipped**. All workflow-equivalent
model/dataset manifest, catalog and example policy checks passed. Logs remain
in the new output directory. REVIEW sources remain unapproved; UnAV-100 is
not used. No R2 quality/fusion gate or production promotion follows this result.
