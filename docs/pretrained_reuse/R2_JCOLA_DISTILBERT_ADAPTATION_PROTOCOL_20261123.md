# JCoLA compact BERT adaptation comparison protocol — 2026-11-23

## Question and scope

The previous one-time JCoLA OOD development comparison found that a frozen
four-layer LINE DistilBERT plus a shared option scorer collapsed to the majority
class. This experiment asks one narrower question: does limited task adaptation
of the same budget-compatible four-layer encoder improve source-held-out
performance over a frozen-feature scorer? It does not revisit the public OOD
development records and does not test general Japanese reasoning or multimodal
quality.

## Frozen inputs and no-evaluation-data rule

- Dataset: `osekilab/JCoLA` revision
  `736d9eef3af04bb17e4cf59adf01325973234ff9`, local train JSON SHA-256
  `232787aa7bb11f3c42e455aad59d5f54cf1ea8255e9de79f77840de60e0541bf`.
- Model: LINE Japanese DistilBERT revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`, weight SHA-256
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
- Only the publisher's in-domain TRAIN records are read. Every row is assigned
  to exactly one held-out fold by its article/source identity; normalized
  sentence overlap is rejected. No publisher validation, OOD development,
  test, or sealed-audit data is opened.
- Ten source-held-out folds, source order shuffled with `random.Random(17)`.
  All ten folds contribute one out-of-fold prediction for every train record.

## Fixed comparison

Both arms use exactly the same folds, binary supplied options, prompt, max pair
length 128, seed 17, batch of 16 decisions, five fixed epochs, no early stop,
no checkpoint selection, and unweighted cross entropy.

1. **Frozen baseline:** first four pretrained DistilBERT layers run frozen;
   cache CLS vectors, train `768 -> 128 tanh -> 1` scorer for five epochs with
   AdamW `lr=2e-3`, `weight_decay=1e-4`.
2. **Adapted candidate:** identical first-four-layer model and scorer, with
   layers 0–1 frozen and pretrained layers 2–3 plus scorer trained for five
   epochs with AdamW `lr=2e-5`, `weight_decay=0.01`, gradient-norm clip 1.0.

This changes encoder trainability while preserving the candidate and corpus.
The smaller learning rate is fixed for encoder fine-tuning; it is not a model
selection sweep. No class balancing, threshold, prompt, or feature treatment
is tuned. Report the fixed epoch-5 result and all fold/epoch observations.

The four-layer candidate's pre-quantization text tower plus scorer is
54,010,625 parameters; the existing four-tower accounting is 195,994,241,
including this readout. Adaptation changes weights, not parameter count. This
does not measure a compressed package or satisfy the full quality target.

## Artifacts and execution

Implementation: `scripts/fit_jcola_distilbert_adaptation.py`.
Focused CPU tests cover complete deterministic source grouping and trainable
layer selection. A separate CPU smoke must pass before the full local run.
Output is external under a new attempt directory, with one small safe-tensor
adaptation/readout per fold, all OOF predictions, training histories, effective
config, hashes, runtime and best/final save-reload checks. No artifacts are
written over the earlier JCoLA run or model weights.

Interpretation is development-only because the same training records informed
previous experiments. A positive result does not establish an untouched
generalization result; a negative result only applies to this compact
architecture, training recipe and source-grouped JCoLA task.
