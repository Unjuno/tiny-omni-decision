# BERT-family text-path evidence reconciliation — 2026-11-23

## Decision

Keep pretrained BERT-family encoders in the text-path candidate set, but stop
unstructured architecture sweeps on the already exposed development cohorts.
Evidence supports useful, task-dependent text representations; it does not
establish general supplied-option reasoning, 95–98% text quality, a complete
multimodal model, or a parameter/size-qualified integrated runtime.

The next BERT-family experiment should require a new, rights-cleared, source-
group-disjoint task and compare candidates under the same option-scoring
contract and total resident parameter budget. Until such a cohort is admitted,
the main product blocker remains a clean multi-input task showing that both
streams are needed and that fusion improves on matched unimodal controls.

## Evidence reviewed

The following were checked against the machine-readable receipts in
`C:\CodexArtifacts\pretrained-reuse-r2-local\`. All results below are
development or internal-validation results. Publisher test sets and the
project sealed audit were not opened.

| Task | Candidate/result | What it supports | Limit |
|---|---|---|---|
| Japanese MASSIVE intent, 2,033 validation rows / 2,015 normalized-text groups | Frozen Tohoku Japanese BERT-base: 81.65% Accuracy, 0.6872 NLL, 0.2730 Brier; frozen Ruri-v3-70m: 75.50%, 0.8983, 0.3483 | BERT-base is a stronger representation on this independent-source Japanese intent task. Group-paired bootstrap favored BERT on Accuracy, NLL and Brier. | Checkpoints were selected on this same validation split; 59 intents are represented, support is uneven, and this is text intent classification rather than arbitrary-option reasoning. |
| Japanese JCommonsenseQA, same grouped 2,004-row validation | Tohoku BERT-base plus option scorer: 70.96%; Ruri-v3-70m: 50.70% | A promising Japanese five-choice signal beyond intent classification. | The readout/feature construction differed between systems, so this is not an encoder-only causal comparison. The pinned release has no publisher test. |
| Japanese JNLI, grouped validation | Tohoku BERT-base: 78.10% overall | The same encoder can support paired-sentence classification. | Entailment accuracy was only 24.18% (contradiction 78.44%, neutral 91.00%); overall accuracy hides severe class weakness. |
| BANKING77 | Frozen MiniLM-L3: 89.69%; L6: 91.44%; L12: 90.69% | Small BERT-derived encoders can produce compact, executable text classifiers. L6 was the best balanced fixed candidate in this comparison. | Narrow English intent labels; publisher test was not opened and source person/conversation groups are unavailable. L12 did not improve Accuracy over L6. |
| CLINC150 | Frozen MiniLM-L6 and L12: 95.72% overall Accuracy | High accuracy is possible on a bounded intent taxonomy. | Only 24 validation examples represented OOS; this does not establish robust open-set behavior or broad text Decision quality. |
| Japanese JCoLA OOD-development subset, 685 rows | Tohoku BERT-base and LINE 4-layer DistilBERT both: 73.28% Accuracy, 50% macro Accuracy, MCC 0; both predicted the majority class for every row. | The same family can fail on a different Japanese decision task; reducing depth alone did not solve it. | The run used a CUDA feature path and CC BY-SA data kept local. The official test and sealed audit were not loaded. |
| Japanese JCoLA train-only, 10-source grouped out-of-fold evaluation, 6,919 rows | LINE Japanese DistilBERT frozen four-layer readout: 83.38% Accuracy / 50.00% macro Accuracy / 0.4560 NLL / 0.0316 ECE; adapting the upper two layers: 82.08% / 56.32% / 0.5188 / 0.0977. | Upper-layer adaptation recovered some minority-class discrimination (MCC 0.182; unacceptable recall 17.74%) from a majority-class collapse. | Accuracy fell 1.30 pp, NLL worsened 0.0627, ECE worsened 0.0661; paired source bootstrap intervals favored the frozen arm for Accuracy and NLL, while the macro-accuracy interval included zero. Only ten source clusters were available. This does not justify further tuning on JCoLA or a broad BERT sweep. |
| Typed Decisions Synth internal comparison, 1,200 decisions / 240 state groups | Japanese BERT-base (110.72M total parameters): 53.40% teacher-mode agreement; Light SPLADE (27.62M): 43.40%. BERT-minus-SPLADE Accuracy difference +10.0 pp (paired state-cluster 95% interval +6.60 to +13.40 pp). | BERT contributes semantic signal over the lightweight lexical baseline in this controlled internal task. | The target is agreement with the synthetic teacher distribution, not independent factual correctness. The validation selected readouts; this is not a product benchmark. |
| Open-Jev amount-control internal split, 6,696 validation rows / 27 document families | Tiny ModernBERT-CJE: 67.57% macro Accuracy, 0.7087 NLL; MiniLM-L3: 69.81%, 0.6778. | A compact pretrained text encoder can operate under a small parameter budget. | Every candidate pair was truncated at the 512-token cap (maximum untruncated length exceeded 1,000 tokens). The result is confounded by input truncation and is not evidence of an encoder ceiling. Labels are programmatically derived and validation groups came from publisher TRAIN. |

The Japanese BERT-base MASSIVE run used the pinned `tohoku-nlp/bert-base-japanese-v3`
revision `65243d6e5629b969c77309f217bd7b1a79d43c7e` (Apache-2.0), with
111,207,168 frozen encoder parameters and 106,172 readout parameters. Its
447,406,217-byte FP32 weight file alone exceeds the aspirational package-size
target. No quantization, latency-qualified runtime, or full multimodal
integration was verified in these BERT comparisons.

### JCoLA upper-layer adaptation follow-up

A separate seed-17, ten-fold source-held-out run used only the publisher's
JCoLA TRAIN records. Every row received one out-of-fold prediction; no
publisher development/test or project sealed-audit records were loaded. It
compared the same four-layer LINE Japanese DistilBERT feature path with a
fixed scorer against an arm that trained only the upper two DistilBERT layers
and scorer. The adapted arm had 14,274,305 trainable parameters within a
54,010,625-parameter candidate; its fixed epoch-5 endpoint was used in every
fold, without validation-based checkpoint selection.

Adaptation improved macro accuracy from 50.00% to 56.32% and MCC from 0 to
0.182, but lowered sample accuracy from 83.38% to 82.08%, worsened NLL from
0.4560 to 0.5188, and worsened ECE-15 from 0.0316 to 0.0977. A paired
10,000-replicate source-cluster bootstrap gave adapted-minus-frozen intervals
of `[-0.0281, -0.0032]` for Accuracy, `[-0.0393, +0.0040]` for macro
Accuracy, and `[+0.0390, +0.0795]` for NLL (positive NLL is worse). With just
ten source clusters, these are diagnostic, not broad population guarantees.
The run used 17.8 minutes and peaked at about 0.90 GiB CUDA allocation; this
completed before the separately observed Orbitune GPU job. The source is
CC-BY-SA-4.0 and its data/artifacts remain outside Git pending product-rights
review.

This follow-up refines the earlier JCoLA collapse finding: partial adaptation
can recover some minority recall, but did not improve the primary accuracy or
probability-quality measures. It is not evidence that further JCoLA tuning
will help the product task, and this exposed cohort should not be revisited.

## Raw-text JSON runtime replay

The pinned model and selected readout were then loaded on CPU through the
raw-text runtime. This was deterministic inference replay, not a new fit or
checkpoint-selection pass:

- Model revision: `65243d6e5629b969c77309f217bd7b1a79d43c7e`, Apache-2.0.
- Model weights: 447,406,217 bytes, SHA-256
  `e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4`.
- Selected readout: 426,616 bytes, SHA-256
  `49d22f13ac6a4b4f3f31798d70d986d815950eb4e4460cb30cc34ab7e012254b`.
- Runtime script SHA-256:
  `b5e91c28b1fcf4ae040d93bebb18d6e68bfd293c8ecb2ea02cf5a29789da7412`.
- Feature encoder helper SHA-256:
  `e2e7a51d90c2f308528b23c3ccf6c10bcc1c809118b0a773e330701d9441131e`.
- Fixed validation JSONL SHA-256:
  `aa42d55cb8afcf7170483322e6a129acd4657a9acf3ea3530fea265a36679707`.
- Saved predictions SHA-256:
  `8c675c3afe1893bec47b53214678935964d4e06931e4a3698a3abc2772319040`.

The exact Python module invocation used was:

```powershell
python -m scripts.runtime_massive_ja_bert_json `
  --model C:\CodexArtifacts\pretrained-reuse-r2-local\tohoku-bertbase-v3-runtime-20261123-attempt-01 `
  --checkpoint C:\CodexArtifacts\pretrained-reuse-r2-local\massive-ja-jp-bert-readout-20261123-attempt-01\best-readout.pt `
  --validation-jsonl C:\CodexArtifacts\pretrained-reuse-r2-local\massive-ja-jp-grouped-split-v1-attempt-01\validation.jsonl `
  --expected-predictions C:\CodexArtifacts\pretrained-reuse-r2-local\massive-ja-jp-bert-readout-20261123-attempt-01\validation-predictions.jsonl `
  --receipt C:\CodexArtifacts\pretrained-reuse-r2-local\massive-ja-jp-bert-json-runtime-replay-20261123-attempt-01\replay-verification.json
```

The replay passed on CPU: **2,033/2,033 argmaxes matched**, maximum probability
absolute difference `0.0` (tolerance `1e-7`, relative tolerance `0`). It took
27.7 seconds with four threads under Python 3.11.9, PyTorch 2.6.0+cu124, and
Transformers 5.6.2. The receipt SHA-256 is
`c2b1d04a0c7e21e0743e6d495a3019c88394e276c8c7fbdef86f3d58376daa2d`.

This confirms one fixed Japanese intent taxonomy from raw text to normalized
probability JSON and reproducible development predictions. It does not support
arbitrary supplied options, another modality, an independent quality estimate,
or a product-level accuracy claim. The validation cohort remains selection-
affected and must not be used for further model selection.

The runtime script was an uncommitted file in a separate existing development
worktree and is identified here by hash; it is not part of this PR. Thus this
receipt proves a local execution, not a clone-and-run implementation delivered
by this branch. The runtime source and external model/readout files still need
to be carried into a tracked, product-scoped implementation before this is
counted as a completed runtime deliverable.

## Reproducibility receipts

These large artifacts remain outside Git; the hashes below identify the exact
JSON receipts inspected for this reconciliation.

| Receipt | SHA-256 |
|---|---|
| `typed-decisions-ja-bert-base-full-fixed-20261123-attempt-01/comparison-summary.json` | `37e44d35ac8c58c0abd1fbfe47442f54c5daa66c25aec51159b976fa53d8a390` |
| `typed-decisions-ja-light-splade-full-fixed-20261123-attempt-02/comparison-summary.json` | `150faea9617898a26344fe493bb47ab7851037a4b647f41825b0ece6fdb08546` |
| `typed-decisions-ja-bert-vs-light-splade-paired-bootstrap-20261123-attempt-01/paired-bootstrap.json` | `78816392655d4bd2378386a853e70cf86d28c0562832c46575b58f7b14280a0d` |
| `jcola-bert-grouped-decision-20261123-attempt-01/run-manifest.json` | `3771fe87201048e5829b358a7f52fff3fdf66c0f86f413ad1dca27abcfe12cd9` |
| `jcola-bert-grouped-decision-20261123-attempt-01/saved-prediction-analysis.json` | `2ca0af27bb5328dac346643b1762950ff0da605080914c7348625700583692ef` |
| `jcola-distilbert-adaptation-cv-20261123-attempt-02/analysis-summary.json` | `cb7f889a12f0e14d89d8ef337559bf68d63154aa3f1c193dedab829eec1b490e` |
| `openjev-amount-modernbert-comparison-20261123-attempt-01/run-report.json` | `9b7810073f8c46f9057980d64580b0bf717b4d91efccb36fae804443a25154cf` |
| `openjev-amount-modernbert-comparison-20261123-attempt-01/paired-family-bootstrap.json` | `2ac293aa9108bf5df1700d01cbd9e9e7d84ccd9d8f708b140422d392dae533a1` |

The MASSIVE and Japanese readout probes are separately documented in their
development branch. Their pinned dataset/model revisions, split hashes,
prediction hashes, reload checks and selection limitations must be retained
when those records are brought into the roadmap branch. This note does not
copy corpora, model weights, predictions, or checkpoints into Git.

## Next action

1. Keep MiniLM-L6 and Japanese BERT-base as task-specific text references;
   do not promote either as the Omni model.
2. Do not retune the exposed BANKING77, CLINC150, JCommonsenseQA, MASSIVE,
   JNLI, JCoLA, or current Open-Jev internal splits.
3. Admit one independent text Decision task only if its source rights, group
   identities, option semantics, and untouched evaluation partition are
   verified before training. Compare a compact BERT-family encoder and the
   current non-BERT text path using an identical readout protocol.
4. In parallel, continue the roadmap's R2 multi-input admission gate. A text
   result cannot close audio/image/video/sensor quality or fusion requirements.

No model was trained and no GPU was used for this reconciliation. One pinned
CPU-only raw-text inference and a fixed development-prediction replay were run;
no sealed audit or publisher test was used.
