# CommonsenseQA BERT-Mini final-layer adaptation — 2026-11-26

## Protocol and status

This was one controlled capacity step after the frozen-feature result. The
matched comparison uses the same pinned BERT-Mini base, train-only shard,
question-concept split, seed, tokenizer, max length, option order, target
semantics, head initialization, head learning rate, batch size, weight decay,
gradient clipping, and validation selector. The only model-capacity change in
the matched run was unfreezing BERT's final encoder layer; that layer used a
separate `2e-5` learning rate while the same shared head used `0.01`.

Both selected checkpoints are at epoch 3. The frozen run continued to epoch 11
under its fixed patience rule, but its best checkpoint remained epoch 3. The
adapted run completed exactly three epochs. Validation IDs and order were
identical. This is a same-development-validation comparison; validation was
used for checkpoint selection and has now been observed, so it is not blind
evaluation.

## Matched results

| Metric | Frozen BERT-Mini + head | Final-layer adaptation | Difference |
|---|---:|---:|---:|
| Accuracy, alias-aware | 31.79% | 31.40% | -0.39 pp |
| NLL | 1.5360 | 1.5252 | -0.0108 |
| Brier, sum over five choices | 0.7697 | 0.7644 | -0.0054 |
| ECE, 10 equal-width bins | 0.0275 | 0.0115 | -0.0160 |
| Mean confidence | 0.3392 | 0.3159 | -0.0233 |
| Macro-concept Accuracy | 32.72% | 32.97% | +0.25 pp |
| Macro-concept NLL | 1.5197 | 1.5093 | -0.0104 |
| Macro-concept Brier | 0.7631 | 0.7575 | -0.0056 |

The adapted model did not improve sample Accuracy. It modestly improved NLL,
Brier, and ECE, with a slight increase in macro-concept Accuracy. This mixed,
small single-seed change does not warrant more BERT-family variants or
promotion. Both models remain far below the dataset's reported human score and
the product aspiration. The result suggests calibration/readout changes may
help slightly; it does not isolate whether the remaining limit is encoder
capacity, representation quality, task/data coverage, or benchmark difficulty.

At the selected adapted epoch, train Accuracy/NLL were 34.55% / 1.4924 versus
validation 31.40% / 1.5252. This is a modest gap, not evidence of severe
overfitting. The best and final checkpoint files are byte-identical because
the selected checkpoint was the final epoch.

## Preserved pilot attempt

An earlier pilot run (`run-01`) reached 31.79% Accuracy / 1.5276 NLL / 0.7658
Brier. It also changed scorer learning rate, batch size, and option shuffle
relative to the frozen baseline, so it is explicitly **confounded** and is not
used to attribute improvement to final-layer adaptation. Its output remains
unchanged outside Git at
`C:\\CodexArtifacts\\pretrained-reuse-r2-local\\commonsenseqa-bert-mini-last-layer-20261126-run-01\\`.
The matched run below is the only basis for the capacity comparison.

## Run record

- Run: `commonsenseqa-bert-mini-last-layer-v1`, output
  `C:\\CodexArtifacts\\pretrained-reuse-r2-local\\commonsenseqa-bert-mini-last-layer-20261126-run-02\\`.
- Source commit: `443b953ef9a977d47a5d190ae74c6578dc44e422`.
- Model: `google/bert_uncased_L-4_H-256_A-4`, revision
  `387825ce42dbb39b87911cdf8e383ee3b25184f8`; base has 11,170,560 parameters.
- Trainable: final transformer layer 789,760 + scorer 257 = 790,017; resident
  text-only total 11,170,817 parameters.
- Seed 17; exactly 3 epochs; batch 64 questions; 375 updates; 7,951 unique
  train questions, 23,853 question presentations and 119,265 option-pair
  presentations. Selected checkpoint consumed 23,853 question presentations.
- Frozen parameters remained bit-identical by pre/post tensor hash. Safe reload
  reproduced validation logits exactly (`max abs error = 0`). All 48,705
  candidate pairs fit the 128-token limit without truncation.
- Local CPU only; CUDA was not initialized; Python 3.11.9, PyTorch 2.6.0+cu124,
  Transformers 5.6.2. Wall time 1,156.45 s, process CPU time 3,757.63 s. No
  cloud compute.
- Effective config SHA-256:
  `e9f549a132bb9ae99f1b6c5b2fa8d10e12e1719b75c3142a177af91395c135b9`.
- Training script SHA-256:
  `2b67017f78c3c330f460013079c252cc05202e79566745b39364417df045c0e7`.
- Best/final adapter SHA-256:
  `9cbf404e703684d1f731cf025a9d2eb4f397c0fa129130a71023d2e59016a871`.
- Independently recomputed prediction metrics match run metadata; prediction,
  history, metadata hashes are retained in the external `run-verification.json`.
  No weights or data rows were committed.

## Decision

Stop this BERT-Mini capacity branch here. The new grouped dataset enabled an
honest task test, but limited train size and low task accuracy do not support a
claim about text Decision quality at 95%, much less four modalities. Do not
open the publisher validation/test or reuse this development split for more
model selection. The next useful work is to improve the task/data portfolio
with new source-disjoint, rights-admitted examples and modality-specific
quality checks, then compare representations on a fresh frozen split.
