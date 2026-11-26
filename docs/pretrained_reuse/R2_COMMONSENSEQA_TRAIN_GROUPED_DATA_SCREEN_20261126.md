# CommonsenseQA train-only grouped data screen — 2026-11-26

## Decision

**Admit for one controlled English text Decision experiment on a fresh
question-concept-grouped split, with duplicate-option semantics recorded.**
Do not treat the result as a four-modality product pass or as a route to 95%
accuracy. The published task paper reports approximately 89% human accuracy,
so 95% on this benchmark is not a defensible expectation for this data.

Only the publisher training partition was downloaded and read. The published
validation and test shards were not downloaded or opened. The selected
validation below is a deterministic 20%-by-concept assignment drawn solely
from the publisher training rows.

## Pinned source and terms

- Dataset repository: `tau/commonsense_qa`, revision
  `94630fe30dad47192a8546eb75f094926d47e155`.
- File: `data/train-00000-of-00001.parquet`, 1,247,103 bytes, SHA-256
  `b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b`.
  The pinned repository tree reports file object ID
  `808dfa3c76503733398a5e6af3715e039cad3127`.
- Official dataset card marks the release MIT and identifies the five-choice
  task, crowdsourced questions, 9,741 train / 1,221 validation / 1,140 test
  rows. The original author independently confirmed MIT in the official
  repository's license issue. Preserve author citation and ConceptNet source
  attribution. This records the published grant and is not a legal opinion.
- The paper describes human-authored questions built from ConceptNet
  source-concept/relation groups, with one intended answer and four
  distractors. It reports 88.9% human accuracy and 55.9% for its BERT-large
  baseline. This is a challenging text commonsense task, not a multimodal
  quality claim.

References: [pinned dataset card](https://huggingface.co/datasets/tau/commonsense_qa/tree/94630fe30dad47192a8546eb75f094926d47e155), [author license confirmation](https://github.com/jonathanherzig/commonsenseqa/issues/5), and [dataset paper](https://aclanthology.org/N19-1421/).

## Data-quality findings

| Check | Result |
|---|---:|
| Train rows / distinct IDs | 9,741 / 9,741 |
| Distinct normalized question concepts | 2,151 |
| Duplicate normalized questions | 1 row |
| Duplicate normalized full MCQs | 0 |
| Rows with repeated normalized option text | 122 (1.25%) |
| Rows where the correct option text has an alias label | 42 (0.43%) |
| Rows with repeated distractors only | 80 (0.82%) |
| Answer-label counts A/B/C/D/E | 1,909 / 1,973 / 1,946 / 1,985 / 1,928 |
| Mean correct / distractor word count | 1.503 / 1.525 |

All rows have unique IDs, five labels A–E, five nonempty choice strings, and a
valid answer key. The 42 correct-text alias rows contain semantically identical
options with different labels. They will not be silently dropped or treated as
unique semantic choices: a probability target should divide its mass equally
among labels carrying the same normalized correct text. The 80 duplicate
distractor rows remain in the data and are reported. The answer-word-length
means are close, but this check alone does not rule out other shortcuts.

## Frozen development split

Split seed: `commonsenseqa-train-question-concept-split-20261126-v1`.
Each normalized `question_concept` is assigned by the first 64 bits of
SHA-256 over `seed + NUL + concept`, modulo 100; buckets 0–19 go to validation.
This rule is fixed before any model run.

| Split | Rows | Concepts |
|---|---:|---:|
| Train | 7,951 | 1,711 |
| Validation | 1,790 | 440 |

Cross-split overlap is zero for normalized question concepts, normalized
questions, and normalized question-plus-option sets. Validation IDs in
source-row order hash to
`32ec6ae3b5a4135e7890d307a61a75dd763471aa5cd34f860cf8512cde1aa57a`; train
IDs hash to
`296189960a4935ac0bfaf850d1ace79451b25fcf1f6adcf8516690e717a67770`.
The exact receipt and source data remain outside Git at
`C:\\CodexArtifacts\\pretrained-reuse-r2-local\\commonsenseqa-data-screen-20261126\\`.

The largest single concept group has 277 rows; this is retained intact in one
split. Concept-grouping makes this validation more conservative than the
publisher's random split, but concepts in answer options can still recur
across groups. No test or sealed-audit data was touched.

## BERT decision

This is suitable for **one** targeted BERT-family reference, not a broad sweep.
The first compact candidate to audit is Google's four-layer, 256-hidden BERT
Mini (`google/bert_uncased_L-4_H-256_A-4`, Apache-2.0, approximately 11.3M
parameters), which is small enough to evaluate as a text tower in the intended
model budget. Its exact checkpoint revision, safe loading, tokenizer behavior,
actual parameter count, and local runtime must pass R1 before training.
Compare it with the existing Path A text representation on the same frozen
7,951/1,790 split, with identical option-scoring and option-order controls.
Use alias-aware target probabilities and report Accuracy, NLL, Brier, ECE,
macro/per-concept results, trainable parameters, CPU/GPU time, and calibration.

The score remains narrow evidence about English commonsense MCQA. The dataset's
human ceiling and source-concept scope mean it cannot validate the requested
95–98% target across text, image, audio, video, or sensor inputs.

## Reproducibility and tests

- Audit: `scripts/audit_commonsenseqa_train.py`; it takes one training shard
  only and rejects validation/test filenames. Its JSON receipt contains no raw
  question or option text.
- Tests: `tests/test_audit_commonsenseqa_train.py` covers deterministic
  concept-group assignment, cross-split exclusion, malformed choices, duplicate
  IDs, and equal mass for semantically identical correct options.
- Focused verification: Ruff passed; pytest `4 passed`.
- No BERT checkpoint was downloaded or loaded, and no model was trained in
  this screen.
