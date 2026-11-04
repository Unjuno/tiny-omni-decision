# QASC evidence-context frozen-feature cross-validation — 2026-11-04

## Scope and decision

This is a bounded **Path A text-only diagnostic**. It tests whether providing
QASC's two supplied support facts changes a frozen MiniLM-L3 Decision scorer
relative to the existing question-only QASC TRAIN-only grouped cross-validation
(CV). The only intended input change is adding `fact1` and `fact2` to the state
field. Encoder, feature construction, scorer, seed, optimizer, epoch count,
fold membership, and checkpoint rule are held fixed.

This is not evidence of general text understanding, multimodal quality, a
product-ready Decision model, or progress through the R2-M synchronized
multimodal gate. The label is unusually close to the provided evidence: the
exact normalized target-option string occurs in one of the two support facts
for 6,167 / 8,134 rows (75.82%). This is a strong answer-span shortcut risk,
even though there is no cross-fold leakage. The result must be read as
evidence-conditioned QASC behavior only.

## Data, model, and controls

- Source: QASC official `train` Parquet only, revision
  `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, CC BY 4.0.
- Source Parquet SHA-256: `b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1`.
- Prepared evidence TRAIN JSONL SHA-256:
  `6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b`.
- Preparation receipt: `C:\CodexArtifacts\pretrained-reuse-r2\qasc-evidence-context-20261104-attempt-01\prepared\preparation-report.json`.
- Evaluation/training rows: 8,134 QASC TRAIN records, 8 supplied options each.
  Official validation, test, sealed audit, and any other split were not loaded.
- Frozen encoder: `sentence-transformers/paraphrase-MiniLM-L3-v2` at
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
  The 17,389,824 encoder parameters were frozen.
- Decision readout: 196,865 trainable parameters; 17,586,689 combined encoder
  plus readout parameters. Per-option frozen features are the same four-way
  concatenation `[context, option, abs(context-option), context*option]` as the
  question-only baseline.
- Seed 17, five deterministic folds, 8 epochs per fold, batch size 64
  questions, AdamW learning rate `1e-3`, weight decay `1e-4`; fixed final-epoch
  checkpoint, no held-fold checkpoint selection. The baseline and evidence
  condition use exactly the same ID-to-fold assignment.
- Group separation joins fact-pair, normalized-question, and
  normalized-question-plus-options identities. The five fold audits report
  zero overlap for each identity type. There are 6,638 independent connected
  components (7,139 fact pairs).
- The maximum evidence-context length is 81 tokenizer tokens; no row exceeds
  the 256-token limit.
- CPU-only execution, four Torch threads. The RTX 3080 was not used by this
  experiment; no cloud compute was used.

## Results

All predictions below are out-of-fold predictions over official QASC TRAIN
only. The question-only reference is the preserved same-fold OOF result.

| Condition | n | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Question only | 8,134 | 0.3535 | 1.9197 | 0.8112 | 0.1727 | 0.5262 |
| Question + two support facts | 8,134 | 0.7799 | 0.6598 | 0.3165 | 0.0342 | 0.8036 |
| Uniform 8-option reference | 8,134 | 0.1250 | 2.0794 | 0.8750 | — | 0.1250 |

The fixed answer-position-prior control on each held fold was 12.9% Accuracy
with about 2.079 NLL. Evidence-conditioned performance by exact target-string
overlap was:

| Target string occurs in either support fact | n | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Yes | 6,167 | 0.7991 | 0.6011 | 0.2901 | 0.0315 |
| No | 1,967 | 0.7199 | 0.8437 | 0.3990 | 0.0551 |

A paired bootstrap over the same saved OOF predictions, clustered by the
6,638 connected components used to keep fact-pair/question/content identities
together, used 10,000 resamples and seed 17. Candidate-minus-question-only
95% percentile intervals were `[+0.4133, +0.4396]` for Accuracy and
`[-1.2984, -1.2220]` for NLL. This is a descriptive interval on TRAIN-only
cross-validation predictions, not an independent test interval and not a
correction for training-set dependence.

The two CPU executions used identical frozen-feature cache bytes and all five
fold checkpoint hashes. Their saved predictions matched exactly for IDs,
targets, predicted options, and probability vectors. Two `target_in_fact`
booleans differed because the first run searched the combined state string,
which can create a substring across the fact boundary. The second run checks
each fact separately and agrees with the preparation receipt. The first run's
report and outputs are preserved; its `target_fact_overlap_strata` and
fact-pair bootstrap fields are superseded by its saved
`analysis-corrigendum.json`. No model was retrained for that correction.

## Run record and artifacts

The reproducible corrected run is in:

`C:\CodexArtifacts\pretrained-reuse-r2\qasc-evidence-context-20261104-attempt-01\run-02-repro\`

- Report SHA-256: `d130268674d953e7fabc0e7cd7416134115dd5e80b56f9a5c46238463f526c45`.
- Effective-config SHA-256: `72bce653b8ec506e5a954d37639d4f276525b139e16a572b284fd1b2f3a6f755a`.
- OOF prediction SHA-256: `a70b0d669a5ae7b12a880c3e8a220218b6f6711c77312dd50970b47f53927dff`.
- Frozen-feature cache SHA-256: `49b849148f663db8c487d44541596833c6fcf5e1b8a26927e5eba9e8894bcd1e`.
- Analysis-corrigendum SHA-256:
  `1d7167b1701b4ad4c3f5d8c05418c10ef7551a7757550848afd972169f1661d2`.
- All five fold checkpoint hashes and exact save/reload probability checks are
  recorded in `report.json`; reload equality passed for every fold.
- 4,080 optimizer updates; 260,288 training-example presentations across the
  five fixed folds; 8,134 unique source rows. Feature extraction took 68.06 s,
  scorer training 28.04 s, and total reported wall time was 97.05 s.
- Run source HEAD was `8cc84688ff9f2d7c39c6a92b51925ae27bf92017`.
  The executed run-02 script SHA-256 is
  `f9daadca2c5faaed358b5836931435de7e8a6a94e98fa3323cc11fc8ff628689`;
  `report.json` records the same value as `runner_sha256`. The checked-in
  runner and analysis auditor retain the corrected implementation. No weight
  or media file is added to Git.
- After this run, CI exposed that the research runner imported optional Torch/
  Transformers dependencies while tests only need its CPU-safe split and
  integrity helpers. Those imports now occur only inside the execution entry
  point. This changes module import behavior, not the recorded training math;
  the run-time script hash above remains the exact hash retained in `report.json`.

## Interpretation and next gate

Adding the supplied support facts raises QASC TRAIN-only grouped OOF Accuracy
by 42.65 percentage points and substantially improves calibration. Much of
this task permits direct answer-span matching, so the result does not establish
that MiniLM contributes a general world representation; the no-exact-span
stratum remains an in-domain, TRAIN-only diagnostic. It does show that the
question-only probe omitted a highly informative task input.

Do not select a product model, tune on QASC official validation/test, or claim
the 95% per-task target from this result. This text-only result does not admit
another synchronized audiovisual task, close R2-M, justify R3 Lens training,
or change the current architecture path. Continue seeking a rights-cleared,
independently grouped joint task with frozen labels and both-input controls;
the latest R2 candidate matrix still reports that gate open.
