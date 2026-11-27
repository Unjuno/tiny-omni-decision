# BANKING77 MiniLM one-shot publisher-test result — 2026-11-27

## Purpose and one-shot boundary

This is a single inference-only evaluation of the already selected
`banking77-frozen-minilm-l3-seed17-attempt-01` checkpoint. The checkpoint,
tokenizer revision, preprocessing, and label vocabulary were frozen from the
publisher-train-derived grouped validation result before opening the official
publisher test. No model selection, retraining, calibration, filtering, or
label changes occurred. This test is now exposed and must not be reused for
tuning. The project's separate sealed audit was not accessed.

The test contains 3,080 rows, exactly 40 per intent. The pinned source is
PolyAI-LDN/task-specific-datasets at rights revision
`57ec275d8078af65b7731c2a98be812d844a6d6b`, file
`banking_data/test.csv`, 239,961 bytes, SHA-256
`d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`. This
matches the published BANKING77 test-file checksum recorded in the [dataset
card history](https://huggingface.co/datasets/PolyAI/banking77/commit/a00bafa8d1db84baced2c9032e3f227f9c486cb2).
The [publisher repository at the pinned revision](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
and the CC BY 4.0 source/provenance limits are described in the
[train-only screen](R2_BANKING77_TRAIN_DATA_SCREEN_20261127.md).

## Integrity checks

- The 17,389,824-parameter frozen MiniLM-L3 encoder was loaded from revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; its weights hash was verified.
- The 29,645-parameter readout was loaded by safetensors and its hash matched
  the selected-run receipt.
- The test file hash and 3,080-row, `text,category` schema were verified.
- All 77 expected labels were present, with 40 examples each.
- 25 test rows (0.81%) have an exact normalized-text match in publisher train.
  They remain included in the official full-test score; no rows were filtered.
- CPU inference only; no GPU or cloud compute. The selected readout and test
  file remain separate from Git.

## Results

| Metric | Train (selected readout) | Grouped validation | Official publisher test |
|---|---:|---:|---:|
| Rows | 7,903 | 2,100 | 3,080 |
| Sample Accuracy | 98.28% | 92.24% | 89.94% |
| Macro Accuracy | 98.40% | 92.23% | 89.94% |
| Sample NLL | 0.1014 | 0.3016 | 0.3735 |
| Macro NLL | 0.0955 | 0.3172 | 0.3735 |
| Multiclass Brier | 0.0375 | 0.1177 | 0.1514 |
| Macro multiclass Brier | 0.0348 | 0.1179 | 0.1514 |
| Top-label ECE (15 bins) | 0.0539 | 0.0157 | 0.0095 |
| Macro classwise ECE | 0.0722 | 0.1174 | 0.1506 |
| Mean confidence | 92.90% | 91.97% | 89.82% |

Official-test Accuracy is 2.29 percentage points below grouped validation and
8.46 points below train macro Accuracy. NLL and Brier also worsen from
validation to test. Top-label ECE is small, but macro classwise ECE is 0.151;
the aggregate calibration number hides substantial per-intent variation.
Each intent has only 40 test examples and there are no user/session IDs, so no
source-group confidence interval is claimed.

Lowest per-intent test Accuracy includes `pending_transfer` 67.5%,
`balance_not_updated_after_bank_transfer` 70.0%, and `declined_transfer` 77.5%.
These overlap with high-frequency validation confusions among transfer,
pending, and balance-update intents. This is consistent with a fine-grained
label-boundary/data challenge, but does not establish that the source labels
are wrong.

## Interpretation

The frozen compact BERT-family text path is a promising representation for
this narrow 77-intent banking task, but the one-shot test result is below 90%
and well below the project's 95–98% aspiration. The grouped validation
overstated official-test performance by 2.29 points, and the train/test gap is
large. This supports treating data provenance, intent-boundary clarity, and
generalization as material issues before testing a larger BERT or adapting the
encoder. It does not show that additional data or a different architecture
alone will solve them.

Do not tune on this public test, drop the 25 overlapping rows, or change source
labels. A future model comparison needs a new untouched task/source cohort with
predefined source groups and independent labels. The source still lacks
customer/session groups; exact-text deduplication is not a substitute for that
metadata.

## Reproduction artifacts

- Evaluator: [`evaluate_banking77_frozen_minilm_test.py`](../../scripts/evaluate_banking77_frozen_minilm_test.py).
- External directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-frozen-minilm-l3-seed17-attempt-01-official-test\`.
- Evaluation report SHA-256:
  `4fa41ba509d1e3a46f34b31e5a48054f47ed812aecee3c60005ccebfd18f6865`.
- External execution receipt SHA-256:
  `1abdc0c9fd764db584ead8946ff78689462fd4a2986e8eb8dd9c568c52d6b956`.
- Prediction file SHA-256:
  `4a66311da1b90460c7bb7056ad14cd0fdba4d938b96f63564ae59fd54a2e3b6e`.
- No test media, sealed audit, or model weights were added to Git.
