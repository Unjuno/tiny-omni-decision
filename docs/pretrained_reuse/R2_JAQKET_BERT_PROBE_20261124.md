# JAQKET frozen Japanese BERT-family probe — 2026-11-24

## Scope

This is a text-only, Japanese 20-choice quiz probe using the supplied candidate titles. It tests whether pretrained Japanese BERT-family features support option-conditioned scoring. It is one narrow knowledge task; it is not a multimodal benchmark or evidence for the 95–98% product target.

The source screen and licensing boundaries are documented in [R2_JAQKET_SOURCE_TASK_SCREEN_20261124.md](R2_JAQKET_SOURCE_TASK_SCREEN_20261124.md). Only the official developer-created Dev1 file was downloaded. The non-commercial competition train set, Wikipedia article-body corpus, Dev2, hidden evaluation data, and sealed audit were not accessed.

## Frozen data and split

Dev1 contains 995 JSONL records. Its SHA-256 is `cef9d57fa479cd4714ebbe67eb5ae6409fcef1d96d4e9f725f5f93a0c6c1124c`. All 995 qids and normalized questions are unique; all rows have 20 distinct options and exactly one target. There are 983 answer entities and 994 unordered candidate sets.

The seed-17 split groups by normalized question, answer entity, and exact unordered candidate set:

| Split | Rows | Groups | SHA-256 |
|---|---:|---:|---|
| Train | 796 | 788 | `e1ef9a97b1136c39fa9fccc50ab8ada72c68bd3883beaa42ef4bfecf764194e8` |
| Validation | 199 | 195 | `17fa4035106ee2598d08de95b530f466ed8b723a9dde924c3288ffe532d691e8` |

Cross-split qid, normalized-question, answer-entity, and unordered-candidate-set overlap is zero. Option-title identity is not disjoint: 832 distinct titles occur on both sides and 33 of 195 validation targets occur as train distractors. Therefore this split measures new questions and mostly new target entities, but it does not establish generalization to an unseen option vocabulary. The most frequent answer-position baseline learned on train is 6.16% on validation; random 20-choice chance is 5%.

The split builder is [build_jaqket_dev1_split.py](../../scripts/pretrained_reuse/build_jaqket_dev1_split.py). External data, split files, model features, readout artifacts, histories, and predictions are under `C:\\CodexArtifacts\\pretrained-reuse-r2-local\\jaqket-choice-probe-20261124-attempt-01\\` and are not stored in Git.

## Frozen comparison

Each candidate title is paired independently with its question. The encoder stays frozen; attention-mask mean pooling feeds one shared `Linear(hidden,128) -> Tanh -> Linear(128,1)` score, then softmax over 20 options. An option-title-only control blanks the question. All six conditions use seed 17, AdamW at 0.002, weight decay 0.0001, batch size 64 questions, a constant learning rate, 100 epochs without early stopping, max length 256, and best-checkpoint selection by minimum validation NLL. All epochs ran; the best and last readout are saved separately. No pair was truncated.

| Encoder / role | Encoder params | Readout params | Pair selected epoch | Pair Accuracy | NLL | Brier | ECE-15 | Option-only Accuracy / NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ruri-v3-30m compact baseline | 36.71M | 33,025 | 5 | 12.06% | 2.820 | 0.932 | 0.068 | 12.06% / 2.849 |
| LINE Japanese DistilBERT first 4 layers, in-budget candidate | 53.91M | 98,561 | 6 | 30.15% | 2.298 | 0.847 | 0.105 | 25.13% / 2.552 |
| Tohoku Japanese BERT-base v3, reference only | 111.21M | 98,561 | 10 | 37.19% | 1.975 | 0.760 | 0.059 | 18.59% / 2.551 |

Option-only Brier/ECE-15 were 0.933/0.026 for Ruri, 0.884/0.060 for DistilBERT, and 0.893/0.028 for BERT-base. Brier is the sum over 20 classes, averaged per question. The encoder and readout remain text-only; audio, image, video, fusion, and calibrated product output were not measured.

### Paired validation diagnostics

A 10,000-replicate paired bootstrap resampled 195 normalized answer-entity groups (seed 1729). The selected checkpoints and model candidates were chosen on this same validation split, so intervals are descriptive and optimistic, not independent test inference.

| Comparison (left minus right) | Accuracy delta, 95% interval | NLL delta, 95% interval | Brier delta, 95% interval | ECE delta, 95% interval |
|---|---:|---:|---:|---:|
| BERT-base pair − DistilBERT pair | +7.0 pp [−0.5, +14.5] | −0.324 [−0.489, −0.164] | −0.087 [−0.136, −0.039] | −0.016 [−0.085, +0.053] |
| BERT-base pair − BERT-base option-only | +18.6 pp [+11.2, +26.1] | −0.577 [−0.736, −0.419] | −0.133 [−0.184, −0.083] | +0.046 [−0.007, +0.099] |
| DistilBERT pair − DistilBERT option-only | +5.0 pp [−2.0, +12.0] | −0.254 [−0.367, −0.140] | −0.037 [−0.068, −0.007] | +0.038 [−0.027, +0.104] |

This supports the claim that BERT-base uses question context on this task: it is materially better than its title-only control. Compared with the in-budget DistilBERT, BERT-base improves validation NLL and Brier, while the Accuracy interval includes zero. ECE differences are uncertain. Ruri changes little when given the question.

### Learning curves and overfit

| Encoder | Best epoch: train CE / validation NLL | Epoch 100: train CE / validation NLL |
|---|---:|---:|
| Ruri | 5: 2.461 / 2.820 | 100: 0.003 / 4.198 |
| DistilBERT first 4 | 6: 2.220 / 2.298 | 100: 0.022 / 4.509 |
| BERT-base | 10: 1.643 / 1.975 | 100: 0.023 / 3.284 |

Train CE kept falling while validation NLL rose after the early selected checkpoint in every condition. This readout-only run therefore shows clear overfitting on 796 questions. It does not show that more epochs help. More licensed, task-relevant questions or stronger regularization are needed before deciding whether to adapt upper BERT layers.

## Size, runtime, and decision

Runs used the local RTX 3080 Laptop GPU (16 GB); no cloud compute was used. Pair-condition peak allocated VRAM was 356 MiB (Ruri), 424 MiB (DistilBERT), and 600 MiB (BERT-base). Option-only peaks were 251, 311, and 515 MiB. The six full runs used about 119 seconds total GPU wall time, excluding smoke runs and data preparation. Training updated only 33,025 or 98,561 readout parameters; encoder trainable-parameter count was zero.

The project’s provisional four-tower parameter projection is about 195.99M for the four-layer DistilBERT candidate and 253.29M for the BERT-base reference. Thus BERT-base is a teacher/reference, not an in-budget product candidate. DistilBERT fits the provisional parameter envelope but reached only 30.15% on this 20-choice validation task. Neither is promoted. The 95–98% target is unmet, and no result here says anything about the other modalities.

### Frozen one-shot public pseudo-test

Before opening Dev2, freeze these six validation-selected readouts by SHA-256:

- Ruri pair: `b9468418c36e7597c8cfede62fe5989fa7f87707a905adfb2e60106068b3287b`
- Ruri option-only: `f1f4baa5d65277aaa90573791f92e1f8ef9e57141c99ac6f98f9f4b88636a4ca`
- DistilBERT pair: `ad8154dff55d69fc31e24457cdb880dc248a0fb68938aaf335b3fd4690b7564b`
- DistilBERT option-only: `06027c4ed828f2f4904b1f6613a2be4a371d98e4ee21d09e6d718cf98be9e384`
- BERT-base pair: `431b895b36089dc625b39fda2feee27adec18df6db8964c04bbf6a44f4a12505`
- BERT-base option-only: `c25d69b969030e60a2c114675113f35819e308b30716ff52c73f7861bd4159b5`

These six readouts were evaluated once on the official Dev2 file (997 rows; SHA-256 `ec4938b0bb19455b69c8770a57f4f5b2baf32311295b3d1557a8bca4152189d9`). No checkpoint or configuration was selected or changed from Dev2. It is a public pseudo-test, not a sealed blind audit.

| Encoder / condition | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Ruri pair | 10.33% | 2.883 | 0.940 | 0.049 |
| Ruri option-only | 11.94% | 2.872 | 0.935 | 0.011 |
| DistilBERT pair | 24.57% | 2.495 | 0.874 | 0.041 |
| DistilBERT option-only | 20.66% | 2.621 | 0.895 | 0.039 |
| BERT-base pair | 38.42% | 2.069 | 0.769 | 0.057 |
| BERT-base option-only | 21.16% | 2.559 | 0.888 | 0.028 |

A one-shot 10,000-replicate paired bootstrap over 989 normalized answer entities (seed 1729) found BERT-base pair vs DistilBERT pair at +13.9 pp Accuracy `[+10.4,+17.4]`, −0.426 NLL `[−0.512,−0.343]`, and −0.105 Brier `[−0.129,−0.081]`. BERT-base pair vs its option-only control was +17.3 pp Accuracy `[+13.9,+20.6]`, −0.490 NLL `[−0.567,−0.413]`, and −0.118 Brier `[−0.142,−0.095]`. ECE differences remained uncertain. These intervals describe this one public pseudo-test split; they do not establish general multimodal quality.

There is no qid, normalized-question, or unordered-candidate-set overlap between Dev1 and Dev2. However, 22 target answer entities overlap (16 were positive train targets, 6 validation targets), and 160 Dev2 targets appeared somewhere in train candidate options. Therefore this is not a fully unseen-entity test. The 95–98% target is still unmet; Dev2 metrics do not change the frozen role assignments or promote any candidate.

## Artifacts and integrity

Each training run has `training-history.json`, `validation-predictions.jsonl`, a separately retained best and final readout, effective config/result metadata, and SHA-256 hashes outside Git. Dev2 predictions and evaluation metadata are also external and hash-recorded in the manifest. [analyze_jaqket_probe.py](../../scripts/pretrained_reuse/analyze_jaqket_probe.py) performs the grouped uncertainty analysis; [evaluate_jaqket_dev2_once.py](../../scripts/pretrained_reuse/evaluate_jaqket_dev2_once.py) locks the evaluation to the frozen readout hashes and refuses an existing output path. Unit tests cover split determinism/integrity, metric calculation, permutation equivariance, one-shot overlap checks, and grouped-bootstrap reproducibility. The original Ruri, DistilBERT, and BERT-base checkpoints were read-only and unchanged.
