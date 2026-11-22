# EVE EO MCQA TRAIN-only grouped CV results

## Outcome

The frozen text encoder plus the existing shared candidate readout scored
65.39% accuracy on 1,257 out-of-fold decisions. This is above the uniform
four-choice reference (27.92%), but far below the 95–98% aspiration and is
evidence for only this narrow, text-only Earth Observation multiple-choice
task. It is not evidence for broad text skill, image/video/audio processing,
multimodal fusion, or product readiness. No source validation/test split or
sealed audit was accessed.

The five outer folds varied from 60.56% to 72.62% accuracy. The task and
checkpoint procedure are now closed; do not tune on these OOF predictions.

## Frozen protocol and provenance

- Source: `eve-esa/mcqa-single-answer`, revision
  `f99c5779ca51e6d3407eeebe8297d5081af8a66b`; pinned TRAIN parquet SHA-256
  `9f711235b688491f27a0b484f6fc31bd3fba1ede261ffc2d500a73f896833509`.
- 1,261 source rows were audited. Four rows (source indices 20, 578, 821, 893)
  have a missing/empty candidate at the position named by the answer label and
  were excluded before feature extraction. No row was removed based on model
  performance. The 1,257 usable rows form 1,257 exact-content groups; no exact
  normalized-question or normalized question-plus-choice duplicates were
  found. Author/document identity and semantic near-duplicate independence
  cannot be established from this source.
- The outer folds have 252, 252, 251, 251, 251 usable examples. Each outer
  training partition has a deterministic 20% group-held inner stop set. Seed
  17; up to eight epochs; earliest minimum inner-stop NLL selects each fold's
  readout. No model/seed sweep was run.
- Frozen encoder: `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`, local weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`,
  17,389,824 frozen parameters. Shared readout: 196,865 trainable parameters.
- Source/config/model/fold hashes and the complete fixed protocol are in the
  [protocol](R2_EVE_EO_MCQA_TRAIN_GROUP_CV_PROTOCOL_20261121.md). The code and
  protocol were committed as `36d303e0fa41df824dbf62506af68dd308f959d9`
  before fitting.
- CPU-only local execution, seed 17, 2026-11-21 18:29:52–18:40:00 JST
  (10m 08s wall time). Frozen feature extraction took 61.55s and outer-fold
  fitting/scoring 2.75s; the remainder was the predeclared 10,000-replicate
  cluster bootstrap. No GPU or cloud compute was used. Environment: Python
  3.11.9, PyTorch 2.6.0+cu124 (CPU device), Transformers 5.6.2, PEFT 0.21.2.

## Metrics

| Outer fold | Scored rows | Selected epoch | Accuracy | NLL | Brier | ECE-15 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 252 | 8 | 72.62% | 0.8139 | 0.3985 | 7.56% |
| 1 | 252 | 4 | 68.25% | 0.8400 | 0.4350 | 9.30% |
| 2 | 251 | 7 | 62.95% | 0.9010 | 0.4650 | 9.10% |
| 3 | 251 | 8 | 62.55% | 1.0580 | 0.4910 | 10.70% |
| 4 | 251 | 5 | 60.56% | 0.9141 | 0.4874 | 4.86% |
| **Pooled OOF** | **1,257** | — | **65.39%** | **0.9052** | **0.4552** | **4.73%** |

The 10,000-replicate seed-17 bootstrap resampled exact-content groups. The
percentile 95% interval was 62.77–68.02% for accuracy, 0.8428–0.9699 for NLL,
and 0.4294–0.4810 for Brier. Since all groups are singletons after exact
deduplication, this is numerically a row-group bootstrap and does not account
for unknown shared source documents/authors.

| Reference | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Uniform over supplied options | 27.92% | 1.3859 | 0.7498 | 2.91% |

The raw result JSON also labels a fold-local deterministic majority answer-
position classifier as an “answer-position prior”: it predicts the most common
training answer position with probability 1. Its accuracy was 27.92%; its
NLL/Brier/ECE are degenerate because all non-modal positions receive zero
probability. Do not interpret those proper scores as a calibrated probabilistic
prior or as evidence against the model; uniform is the comparable probability
reference in this run.

## Learning curves and integrity

Each fold's train CE and inner-stop metrics for epochs 1–8, per-example OOF
probabilities, and selected readouts are retained in the external run folder.
Stop NLL generally fell early; selected epochs were 4–8, with some later
inner-stop NLL regression in folds 1, 2, and 4. Fold 3's selected checkpoint
was epoch 8 despite its weakest proper scores. This variance supports caution;
it does not justify another tune on these same OOF results.

The first preflight attempt stopped before fitting when it encountered an
invalid answer-target candidate. A complete TRAIN-only scan found four such
rows, their fingerprints were recorded, and the frozen input-exclusion rule
was applied before the successful fit. No weights, raw dataset rows, or
predictions were added to Git. Full artifacts are under
`C:\CodexArtifacts\pretrained-reuse-r2\eve-mcqa-train-groupcv-v1\run-01`.

- Result JSON SHA-256: `09379329cef23607ba3109a4cd00db1030bbc6f01bab12194ba6d7cfb9636546`.
- OOF prediction JSONL SHA-256:
  `f27656199c54fa61171dc28158e437672d3c45d54a1550faa356bb3631f45c76`.
- Frozen feature tensor SHA-256:
  `baf47944ae78022800bb682f65d47a836bc293c3cb46bc6419221f453f07e183`.
- Runtime preflight JSON SHA-256:
  `059413cc9a52bb7c6733ee2fa10b9000bd52448546ef01151ca43b145b821e0f`.
- Effective config SHA-256:
  `8f6b05965acfaa8c108840c9e07f3426dfb79f10e95940338edcc7af47fc5444`.

## Decision

Retire this source/task candidate for further tuning. The readout shows useful
task-specific transfer over uniform choice, but the weak and variable result
does not move the 95–98% multi-modality objective materially. Prefer a fresh,
rights-cleared task that tests actual input use and supports a trustworthy
group-held evaluation. HubEgo remains gated: use its payload only after the
user completes and approves access through their account; no access form or
payload was submitted/opened in this run.
