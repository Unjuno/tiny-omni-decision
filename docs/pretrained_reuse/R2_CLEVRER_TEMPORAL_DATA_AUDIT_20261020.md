# CLEVRER temporal coverage data and error audit (2026-10-20)

## Scope

This is a read-only audit of the frozen 3,526-row train / 321-row development
validation corpus and the already selected BiLSTM predictions from the temporal
coverage experiment. It did not load a model or media, train or select a
checkpoint, change data, inspect sealed audit material, or use the audit output
to retune the run. The saved validation predictions are post-hoc diagnostics on
the same development split used for checkpoint selection.

Reproduce with:

```powershell
python -m scripts.audit_clevrer_temporal_readout `
  --corpus-dir C:/CodexArtifacts/pretrained-reuse-r2/clevrer-textfree-temporal-media-v1-attempt-01 `
  --predictions C:/CodexArtifacts/pretrained-reuse-r2/clevrer-textfree-temporal-readout-v1-attempt-01/selected-validation-bilstm.jsonl `
  --output C:/CodexArtifacts/pretrained-reuse-r2/clevrer-textfree-temporal-readout-v1-attempt-01/posthoc-data-audit.json
```

The corpus source hashes are unchanged from the temporal coverage protocol:

| Input | SHA-256 |
|---|---|
| Train JSONL | `282174947f96cb8e33f8ada4226b4eda15a16c82ba4d45656e7cfb64b3958ed9` |
| Validation JSONL | `84f87d52bd612375314602a74295da4af7af2e8142065b47cb1612bd28efca38` |
| Selected BiLSTM predictions | `eebbefcd8b453fa8542f45dbd29a939935077a6cab5ea35c3e140473257561a1` |
| External JSON audit result | `9335df504f9ee76ea8c58e1ebc6d0ea81ae415f5fd89f9d2085f749a80b38a08` |

## Findings

- Integrity checks passed: 324 train scenes and 81 validation scenes, zero
  scene/media/content overlap, and prediction IDs exactly match all 321
  validation rows.
- The fixed `count` option list contains `0` through `5`, but the train subset
  has no positive target `5`; validation also has no target `5`. The readout
  therefore had no positive training evidence for one declared answer and this
  cohort cannot measure that answer. Do not interpret the count result as
  coverage of the full option set.
- Train targets are uneven for count (`0:328, 1:349, 2:264, 3:224, 4:125`),
  while validation has only 3 examples of `4`; the 321-row validation also has
  limited per-taxonomy counts. This increases uncertainty for tail classes.
- The selected BiLSTM validation Accuracy/NLL/Brier are 38.32% / 1.1774 /
  0.6524. The train-frequency prior scores 31.78% / 1.2195 / 0.6635. This
  weak prior comparison is descriptive and does not show that the learned
  representation is sufficient.
- Performance is uneven by taxonomy. `query_color` is 15.52% Accuracy and
  2.1029 NLL over 58 rows; `query_material` is 56.52% / 0.6541 over 46;
  `query_shape` is 50.98% / 0.9787 over 51; `count` is 31.40% / 1.3617 over
  86; `exist` is 43.75% / 0.7360 over 80. These are selected-validation
  diagnostics, not unbiased task estimates.
- Question templates are diverse at the normalized string level (519 train
  templates for count, 372 exist, 350 color, 296 material, 312 shape), but
  common count prompts include “How many collisions happen?” (77) and “How many
  objects enter the scene?” (69). Repeated templates are not split leakage by
  themselves; all questions from each video remain grouped together.
- Program-token subgroup counts overlap because one program may contain several
  temporal operator families. Their metrics are therefore descriptive and do
  not isolate causal operator difficulty. Collision/partner queries are 71/321
  validation rows; entry/exit 209; order 119; frame 162.

## Interpretation and next gate

This audit gives a concrete data-coverage concern in addition to the previously
measured validation overfit and weak video result. It does not establish whether
the low result is caused by feature loss, video reasoning, readout capacity,
label noise, or task construction. The natural-language question is retained
only for split auditing; the trained model received video features, typed
program tokens, taxonomy, and options.

Before another fit on this task, construct a new immutable candidate split that
either contains positive examples for every declared answer class or explicitly
reduces the task contract based on the source's actual label semantics. Keep
whole videos grouped, preserve all original answers/options, and do not
oversample or synthesize examples to fill a missing class. Pre-register the
coverage gate and evaluation counts. Do not tune again on the current
development validation. A separate, grounded task requiring complementary
modalities remains necessary for the product-level Omni objective.

## Implementation and checks

The audit implementation is `scripts/audit_clevrer_temporal_readout.py`, with
regression coverage in `tests/test_clevrer_temporal_readout_audit.py`. Four
CPU tests pass. The runner rejects cross-split scene/media/content overlap,
missing/duplicate prediction IDs, invalid targets, and malformed probabilities.
