# GSM8K-derived MiniLM Decision probe — protocol frozen 2026-11-26

## Scope and claim boundary

This is one exploratory English arithmetic option-selection task, not a
general-text or multimodal quality benchmark. GSM8K was authored as open-answer
math QA; converting each numeric answer into a four-choice Decision is a
project-defined task. The official publisher test split is not read, scored,
or used for selection. No result from this probe can close R2, support a 95–98%
product claim, or qualify a multimodal model.

## Pinned data and split

- Source: `openai/grade-school-math`, commit
  `3101c7d5072418e28b9008a6636bde82a006892c`.
- License: MIT per the dataset card and pinned source repository; preserve
  OpenAI attribution and license notice.
- Only `grade_school_math/data/train.jsonl` was acquired. Its size is
  4,166,206 bytes and SHA-256 is
  `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.
- The file contains 7,473 records with only `question` and `answer` fields;
  every record has an extractable official `####` numeric final answer.
- Exact normalized-question duplicates: 0. A fixed word-trigram SimHash
  screen (64 bits; four 16-bit LSH bands; candidate Hamming distance <=6;
  exact trigram Jaccard >=0.75) found one near-duplicate pair. Those two
  records are assigned to one group. Semantic paraphrase duplicates and
  writer/source identity cannot be ruled out; the source has no author/group ID.
- Seed 17 assigns 6,726 train questions / 6,725 groups and 747 validation
  questions / 747 groups. The split is frozen by sorting SHA-256 ranks of
  `seed=17:<group-id>`; train/validation problem IDs and normalized content
  do not overlap.
- The validation is an internal holdout of the publisher's train split. It
  is not an independent source-writer split or a sealed audit.
- Near-duplicate-connected records share one serialized `source_record_id`;
  each example still has a distinct sample `id`.

## Derived four-choice task

1. Strip the solution and use only the problem question as model input.
2. Parse the final numeric answer after `####` as the correct option.
3. Build a training-only answer bank from unique final-answer values in the
   6,726 train records (822 distinct values).
4. For each question, select three distinct distractors from the 32 training
   answers nearest by relative numeric distance, choosing among that pool by
   a deterministic SHA-256 rank. Exclude the correct value.
5. Shuffle all four numeric options by a deterministic per-question hash.
   Each question contributes one four-way cross-entropy term.

The validation answer is used only to define its correct choice and select
three distractors from the frozen training-only bank. This is a synthetic
choice conversion; it does not reproduce a human-authored multiple-choice
benchmark. Candidate construction, seed, and split may not change after the
validation metrics are observed.

## Model and sole training run

- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- License: Apache-2.0 per its pinned model card.
- Base `model.safetensors`: 90,870,598 bytes; SHA-256
  `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
- Total model parameters: 22,713,601. Update only the existing last encoder
  block and classifier: 1,774,849 parameters. All other parameters remain
  frozen and are hash-checked before/after training.
- Initialize from the pinned pretrained checkpoint, never from the previous
  synthetic-task adapter or its optimizer state.
- Seed 17; three complete epochs; batch size 8 questions; AdamW learning rate
  `2e-5`, weight decay `0.01`, constant schedule, gradient norm clip `1.0`.
  Dropout remains disabled via `eval()` while trainable gradients are enabled.
- Evaluate after the pretrained baseline and each epoch. Select the earliest
  minimum validation NLL. Save selected and final deltas separately.
- Do not add seeds, change the split/options/model, tune a threshold, fit
  calibration, or access the official test based on these results.

## Fixed metrics and controls

Report validation Accuracy, NLL, Brier, ECE-15, mean confidence, and a
uniform-random 25% Accuracy reference for the pretrained and selected models.
Also preserve per-question predictions and probabilities. Run fixed controls
after selection: option-order permutation with scores mapped back to option
text, and a deterministic question-swap ablation against the unchanged options.
Use a 10,000-replicate paired question-group bootstrap (seed 17, percentile
95% intervals) only as a descriptive uncertainty diagnostic; writer-level
independence is unknown. Rolling training CE is not a fixed-example
generalization gap.

## Pre-fit artifacts

- Effective generated split: `C:\CodexArtifacts\pretrained-reuse-r2-local\gsm8k-bert-source-audit-20261126-attempt-01\seed17-four-choice-split-v2`.
- Train JSONL SHA-256:
  `17f4155ceb3763b1d6e97ca8e810d843a66c3b30053baaf1cf8e42c875eb95a1`.
- Validation JSONL SHA-256:
  `e35df851ee67fbb4b2bcfc63df6cb1516731275e75a866ef56cb996de2914340`.
- Split manifest records the builder and train-source hashes. The full raw
  data, predictions, and weights remain outside Git.

