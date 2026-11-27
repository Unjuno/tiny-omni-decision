# R2 BERT and text-data quality recheck — 2026-11-28

## Decision

Continue to treat BERT-family encoders as **task-specific text candidates and
teachers**, not as evidence for a general 95–98% Decision model. Existing
results are mixed across datasets and protocols. Data/task difficulty explains
more of the current spread than a clean model-size comparison can establish.
Do not run another BERT size sweep on the already exposed CommonsenseQA or
Banking77 validation sets. Select a new source-disjoint Decision task and
validate its data/label quality first; only then run a matched BERT-Mini versus
BERT-base comparison if the full-model parameter budget can accommodate it.

This is a read-only audit of pinned local artifacts and committed execution
code. No test or sealed-audit records were opened, and no new model was trained.

## Pinned BERT-Mini audit and CommonsenseQA experiments

The Google BERT-Mini checkpoint is pinned at revision
`387825ce42dbb39b87911cdf8e383ee3b25184f8`. Its local audit recorded
11,170,560 parameters, CPU-only local loading, finite output of shape
`[1, 13, 256]`, and four file hashes. The model card at that revision declares
Apache-2.0. The English CommonsenseQA dataset is pinned at
`94630fe30dad47192a8546eb75f094926d47e155`; its Hub card declares MIT.

The audit used only the 9,741-row publisher train file (SHA-256
`b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b`). Its
seeded `question_concept` split has 7,951 train rows / 1,711 concepts and
1,790 validation rows / 440 concepts. Normalized question, MCQ and concept
overlap are all zero. Answer positions are nearly balanced. Data caveats
include one normalized duplicate question, zero duplicate normalized MCQ
rows, 122 rows with repeated option text, 42 correct-answer aliases, and
eight validation rows with an alias target. No candidate pair was truncated
at the 128-token limit. This is a useful grouped, five-choice task, but alias
and duplicate-option semantics need to remain explicit in any metric.

| Frozen encoder / trainable scope | Parameters | Selected validation result |
| --- | ---: | --- |
| BERT-Mini frozen + 257-parameter scalar option scorer | 11,170,817 total | Accuracy 0.3179; macro-concept Accuracy 0.3272; NLL 1.5360; Brier 0.7697; ECE-10 0.0275 |
| Same BERT-Mini with final encoder layer + scorer trainable | 11,170,817 total; 789,760 encoder parameters trainable | Accuracy 0.3140; macro-concept Accuracy 0.3297; NLL 1.5252; Brier 0.7644; ECE-10 0.0115 |

Uniform five-choice expected Accuracy is 0.2009, NLL 1.6094 and Brier
0.7978. The frozen candidate selected epoch 3 and early-stopped at epoch 11
of 80. The last-layer candidate ran its predeclared three epochs. It slightly
improved NLL/Brier but did not improve sample Accuracy. Both reloaded with
exact logits; the adapted run verified that frozen weights were unchanged.
These are modest task-specific gains over chance, not high-accuracy Decision
results. The official test and sealed audit remain unopened.

The R1 receipt SHA-256 is
`b51ac33eb1a1d86caac87eef6062df6746ad48c345a5ccf72175e3c037504f6b`. The
frozen-run metadata / verification hashes are
`e69c93ea5cc2a7961c6e1c7c07a544b3abf33754fd39da2d3d158879a194ebf8` /
`35e5236918cf60b298cf6e29d6822fa442f21674cfbda46735d871ab4dadfd29`. The
last-layer run-02 metadata / verification hashes are
`100b04a2e02448da0b8105db7c399b5d9f27c756e43f74dd18a3de6202f5a087` /
`d87f2717771dc656df7ce9b510d9dd40434b2105ab9f36a5854d73fa6f9d0ecc`.
Both training source commits and script hashes resolve in Git history.

## Higher Banking77 result is useful but narrower

An existing seed-17 MiniLM-L3 Path A probe used the pinned Banking77 train
source at revision `9d081458ff52e53cf7e848f414e6e9344e4e6696` and grouped
train/validation IDs. The data card/repository license is recorded as
CC-BY-4.0. On its 2,100-row, 77-intent validation set, the selected
17,419,469-parameter model reached 0.9224 sample / 0.9223 macro Accuracy,
0.3016 sample / 0.3172 macro NLL, 0.1177 sample Brier, and 0.0157 top-label
ECE. This is strong evidence for a **closed-set banking intent classifier**,
not arbitrary supplied-option reasoning or multimodal performance.

Data quality remains relevant: the 10,003-row source has 77 intents with
35–187 rows each, 31 normalized duplicate groups, one exact duplicate pair
with conflicting labels, and 57 near-duplicate cross-intent candidate pairs.
The source has no user/session IDs; lexical components were used as a grouping
proxy. Most importantly, the official Banking77 test was previously opened
once for a different MiniLM-L6 candidate. It is no longer a blind evaluation
set for another Banking77 checkpoint and must not be reused for selection.

## How prior BERT results should be interpreted

The existing Japanese BERT-base results are positive but not directly
comparable to the English BERT-Mini probe: they use different languages,
datasets, splits, and in one five-choice comparison different scorer
architectures. The Japanese BERT-base result on five-choice Japanese
CommonsenseQA is 0.7096 validation Accuracy; its source is CC-BY-SA-4.0, so
product reuse requires a separate license decision. MASSIVE is a fixed
60-intent task, not supplied-choice QA; its publisher test has since been
opened for another model and must not be used for another candidate. The
documented BERT-base encoder has about 111M parameters and makes the pinned
four-tower proposal exceed the 200M pre-quantization limit when substituted
for Ruri-30m.

Thus BERT-base is a useful text teacher/reference, while smaller BERT-family
models need a demonstrably good task/data pairing. The current evidence does
not justify further tuning on these exposed splits, an additional benchmark
sweep, or a claim that model capacity alone explains low results. Preserve
the full per-task results and data caveats; prioritize a fresh, rights-cleared,
source-disjoint Decision dataset before the next text training run.

## Sources and artifacts

- Pinned BERT-Mini [model card](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/tree/387825ce42dbb39b87911cdf8e383ee3b25184f8) and CommonsenseQA [dataset card](https://huggingface.co/datasets/tau/commonsense_qa/tree/94630fe30dad47192a8546eb75f094926d47e155).
- Existing BERT cross-task results: [R2_TEXT_BERT_CROSS_TASK_RESULTS_20261123.md](R2_TEXT_BERT_CROSS_TASK_RESULTS_20261123.md).
- External BERT-Mini audit/run outputs: `C:\CodexArtifacts\pretrained-reuse-r2-local\bert-mini-r1-audit-20261126` and `commonsenseqa-bert-mini-{20261126-run-01,last-layer-20261126-run-01,last-layer-20261126-run-02}`.
- External Banking77 data audit and run: `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-train-audit-20261127-attempt-01` and `banking77-frozen-minilm-l3-seed17-attempt-01`.
