# BSD10k class support and validation performance — 2026-11-28

## Question

Does the current weak Path A validation result align with limited training
coverage, and is low example count enough to explain the class failures?

This is a post-hoc descriptive analysis of the already exposed validation used
for checkpoint selection. It is not a new evaluation, a model-selection rule,
or blind evidence. No samples were added, removed, relabeled, or reweighted.
The BSD10k custom test, DCASE evaluation set, and sealed audit remain unopened.

## Method and inputs

- Source state: repository commit `dc4d51018626b7b12eaf7677d320daad6d288834`.
- Readout: the pinned EfficientAT DyMN-0.4 frozen-feature, seed-17 linear
  readout at the existing validation-selected checkpoint (epoch 100).
- Inputs: existing `run-report.json` class metrics and `data_support_by_class`
  counts from
  `C:\CodexArtifacts\pretrained-reuse-r2-local\bsd10k-v1.2-source-audit-20261128-attempt-01`.
- Class observations: all 23 fixed BSD10k classes; predictor metrics are the
  saved temperature-scaled validation accuracies. Support is the frozen train
  count and number of distinct uploader groups. No raw validation media or
  test metadata was opened for this analysis.
- Statistic: Spearman rank correlation across the 23 classes, average ranks
  for ties. Correlations are descriptive; no inferential p-values are claimed.

## Result

| Training support measure | Spearman correlation with validation Accuracy |
| --- | ---: |
| Train examples per class | 0.775 |
| Distinct train uploader groups per class | 0.691 |
| Validation examples per class | 0.594 |
| Distinct validation uploader groups per class | 0.616 |

The direction is consistent with better coverage supporting better class
performance. It does not show that merely adding more clips will cause the
observed differences: class difficulty, taxonomy overlap, annotation ambiguity,
and source acoustics are confounded with support.

Examples make that limitation visible:

| Class | Train clips | Train uploader groups | Validation clips | Validation Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `is-w` | 64 | 14 | 106 | 17.9% |
| `sp-c` | 74 | 43 | 26 | 11.5% |
| `fx-el` | 136 | 26 | 16 | 0.0% |
| `fx-ex` | 111 | 62 | 43 | 7.0% |
| `fx-h` | 453 | 211 | 128 | 85.9% |
| `fx-n` | 352 | 216 | 74 | 93.2% |
| `m-sp` | 343 | 121 | 69 | 82.6% |

`is-w` has both low train support and only three validation uploader groups,
so its estimate is especially uncertain. Conversely, `fx-el` and `fx-ex` have
more training groups but near-zero validation accuracy. For those classes,
reviewing the class definitions, annotation agreement, and error confusions is
at least as important as collecting more rows. Preserve all difficult cases;
do not reduce the label space or tune on the unopened test.

## Decision

Data coverage is a plausible contributor, and independent recording/uploader
diversity is a more meaningful target than repeated epochs over the current
clips. However, the current evidence does **not** justify blindly mapping
FSD50K positives into BSD10k's mutually exclusive 23-way classes: FSD50K
annotations are incomplete and absence is unknown, while several BSD class
names are broad or overlapping. Any cross-source expansion needs a documented
class crosswalk, per-example positive/unknown masks, source/uploader separation,
duplicate checks, and an unchanged held-out evaluation protocol. If a
defensible crosswalk cannot be established, keep FSD50K as a separate task
rather than injecting noisy pseudo-labels into BSD10k.

## Reproduction

The following values were computed from the saved report and split metadata;
the model was not reloaded and no predictions were regenerated:

```text
Spearman(23 class train-example counts, saved validation Accuracy) = 0.775
Spearman(23 class distinct train-uploader counts, saved validation Accuracy) = 0.691
```

Original run hashes, data receipts, model configuration, full per-class metrics,
and test isolation are documented in
[`R2_BSD10K_PATH_A_AUDIO_20261128.md`](R2_BSD10K_PATH_A_AUDIO_20261128.md).
