# CommonsenseQA BERT-Mini frozen-feature result — 2026-11-26

## Result and interpretation

One CPU-only run evaluated a pinned, frozen BERT-Mini encoder and a shared
257-parameter linear candidate scorer on the fixed, question-concept-grouped
split documented in
[`R2_COMMONSENSEQA_TRAIN_GROUPED_DATA_SCREEN_20261126.md`](R2_COMMONSENSEQA_TRAIN_GROUPED_DATA_SCREEN_20261126.md).
The run used only the publisher training shard; no official validation/test or
project sealed-audit data was loaded.

| Metric | Selected validation | Uniform choice baseline |
|---|---:|---:|
| Accuracy, alias-aware | 31.79% | 20.09% |
| NLL, alias-aware | 1.5360 | 1.6094 |
| Brier, sum over five choices | 0.7697 | 0.7978 |
| ECE, 10 equal-width bins | 0.0275 | — |
| Mean confidence | 0.3392 | 0.2000 |
| Macro-concept Accuracy | 32.72% | — |
| Macro-concept NLL | 1.5197 | — |
| Macro-concept Brier | 0.7631 | — |

At the selected epoch, train Accuracy/NLL were 33.63% / 1.5014; validation was
31.79% / 1.5360. The small gap and nearly flat validation curve do not suggest
strong overfitting. The best selector value (macro-concept validation NLL) was
at epoch 3; training continued through epoch 11 under the fixed patience rule.
The frozen representation is above chance but weak for this task. This does
not show that BERT-family encoders generally fail, and does not test a learned
encoder adaptation.

This result is far below the product aspiration. The CommonsenseQA paper
reported 88.9% human accuracy and a 55.9% BERT-large baseline on the authors'
benchmark protocol; those published numbers use a different split and are
context only, not a matched comparison. The dataset is a useful, task-specific
stress test, not evidence that any one modality has passed 95%.

## Model and training record

- Model: `google/bert_uncased_L-4_H-256_A-4`, revision
  `387825ce42dbb39b87911cdf8e383ee3b25184f8`, Apache-2.0 per model card.
- Safe local loading: `torch.load(weights_only=True)` and Transformers
  `AutoModel.from_pretrained(local_files_only=True, weights_only=True)`.
- Actual loaded encoder parameters: 11,170,560; trainable scorer parameters:
  257; total resident parameters for this text-only probe: 11,170,817.
- Tokenizer: fast WordPiece; max length 128; 0 of 48,705 candidate pairs were
  truncated.
- Seed 17; encoder frozen; CLS embedding of each `(question, choice)` pair;
  shared linear scalar scorer; soft target splits mass among identical correct
  option strings; AdamW, learning rate 0.01, weight decay 0.01, batch 64,
  maximum 80 epochs, patience 8, min delta 0.0001; selector: minimum
  macro-concept validation NLL.
- 7,951 unique train questions; 1,790 validation questions / 440 concepts.
  Full run: 11 epochs, 1,375 optimizer updates, 87,461 example presentations.
  Selected epoch 3: 23,853 train example presentations. These counts refer to
  different points in the run and are recorded separately.
- Hardware: local CPU only, CUDA hidden and never initialized. Python 3.11.9,
  PyTorch 2.6.0+cu124, Transformers 5.6.2. Feature extraction took 179.68 s;
  total run wall time 187.15 s and process CPU time 561.31 s. No cloud compute.
- Save/reload maximum logit difference: 0. Option permutation equivariance
  maximum error: 0.

## Reproducibility and artifacts

- Dataset train shard SHA-256:
  `b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b`.
- Model checkpoint SHA-256:
  `01671831dfec6818302153919efbfd6ee4ad91dd9fba7f3649bdd6cb7b0f16e4`.
- Effective config SHA-256:
  `06181e1418b3c4888552ecd466ce2e452b3436f941105e312f40f4b098a3639d`.
- The training script hash at run time was
  `18c0e4b9d79b280a4fef509bda82c027a1e28c27362cd6a0eda1ff74c576d791`; the
  audit script hash was
  `78753d7a1c831abb6e53cc215911515e05a97736d4a643165d2d4d705b4cffb6`.
  The worktree had uncommitted experiment code at execution; these file hashes
  bind the executed contents, and the scripts are included in the follow-up
  repository commit.
- Selected scorer SHA-256:
  `f80400fe65f67d6a70e79f5fb36595ee53d2615a40fa12bab9c9e1ed6a64023f`.
- Full training history, validation ID/prediction/probability records,
  run metadata, and independent verification receipt are outside Git at
  `C:\\CodexArtifacts\\pretrained-reuse-r2-local\\commonsenseqa-bert-mini-20261126-run-01\\`.
  No weights or dataset rows were added to Git.

## Next experiment gate

The frozen-feature fit is insufficient for high quality, while the small train
gap gives no evidence that additional epochs on a frozen encoder will solve
it. One narrow capacity step is justified: keep the same model, tokenizer,
data, split, seed, target, and shared scorer, but unfreeze only the last BERT
layer. Compare against this frozen result and use validation-only checkpoint
selection. Do not expand to additional BERT families unless that result changes
the architecture decision.
