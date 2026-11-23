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
| Typed Decisions Synth internal comparison, 1,200 decisions / 240 state groups | Japanese BERT-base (110.72M total parameters): 53.40% teacher-mode agreement; Light SPLADE (27.62M): 43.40%. BERT-minus-SPLADE Accuracy difference +10.0 pp (paired state-cluster 95% interval +6.60 to +13.40 pp). | BERT contributes semantic signal over the lightweight lexical baseline in this controlled internal task. | The target is agreement with the synthetic teacher distribution, not independent factual correctness. The validation selected readouts; this is not a product benchmark. |
| Open-Jev amount-control internal split, 6,696 validation rows / 27 document families | Tiny ModernBERT-CJE: 67.57% macro Accuracy, 0.7087 NLL; MiniLM-L3: 69.81%, 0.6778. | A compact pretrained text encoder can operate under a small parameter budget. | Every candidate pair was truncated at the 512-token cap (maximum untruncated length exceeded 1,000 tokens). The result is confounded by input truncation and is not evidence of an encoder ceiling. Labels are programmatically derived and validation groups came from publisher TRAIN. |

The Japanese BERT-base MASSIVE run used the pinned `tohoku-nlp/bert-base-japanese-v3`
revision `65243d6e5629b969c77309f217bd7b1a79d43c7e` (Apache-2.0), with
111,207,168 frozen encoder parameters and 106,172 readout parameters. Its
447,406,217-byte FP32 weight file alone exceeds the aspirational package-size
target. No quantization, latency-qualified runtime, or full multimodal
integration was verified in these BERT comparisons.

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

No model was loaded or trained for this reconciliation, and no GPU, sealed
audit, or publisher test was used.
