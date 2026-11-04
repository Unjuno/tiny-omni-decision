# QASC evidence-context TRAIN-only CV protocol — 2026-11-04

## Hypothesis and scope

The prior Path A QASC control scored only `question + candidate option`; the two
publisher-provided supporting facts were deliberately omitted. Its leakage-aware
TRAIN-only 5-fold OOF Accuracy was 35.35%, NLL 1.9197 and ECE-15 0.1727. This
one-factor diagnostic tests whether supplying those two facts as the observation
state improves this narrow grounded science-choice task. It does not reinterpret
the old question-only result, tune on the exposed official validation, or measure
general text or multimodal quality.

## Frozen design

- Source revision: `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, CC BY 4.0.
- Download only `data/train-00000-of-00001.parquet`, expected SHA-256
  `b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1`.
  Do not fetch or inspect validation/test shards.
- Exact 8,134 prepared TRAIN IDs, choices and targets must join one-to-one to
  source rows. Set the Decision `state` to the two supplied facts in source order;
  keep question, options, targets and fact-pair group IDs unchanged.
- Audit fact-pair identity, row-ID order, duplicate facts and whether the gold
  option string appears verbatim in either supplied fact. Such occurrences are
  reported as input-grounding characteristics, not removed.
- Reuse the exact question-only experiment's deterministic seed-17, 5-fold
  assignment. Require the per-fold sample-ID hashes to match the prior artifact,
  with zero fact-pair, normalized-question, and question+choice overlap.
- Same frozen MiniLM-L3 checkpoint (`4ca70771034acceecb2e72475f72050fcdde4ddc`,
  17,389,824 parameters), masked-mean pooling, 196,865-parameter candidate scorer,
  AdamW lr `1e-3`, weight decay `1e-4`, batch 64 questions, eight fixed epochs,
  seed 17 for each fold. Use epoch 8 for every fold; no held-fold selection.
- Same question-option feature structure `[context, option, |difference|, product]`.
  The sole intended model-input change is nonempty state containing the two facts.
- Extract/cache features for TRAIN only; fit and score grouped OOF only. Never
  load the official validation/test labels or sealed audit. Keep the old
  question-only cache, checkpoint and metrics read-only.
- Report Accuracy, NLL, Brier, ECE-15, confidence, fold metrics, answer-position
  baselines, per-fold train CE, save/reload parity, feature-truncation counts,
  parameter counts, runtime and hashes. Compare paired OOF predictions on the
  fixed same folds; present this as internal development CV, not blind testing.

## Interpretation rule

A gain would support the value of explicit evidence in this task, not a general
text-model improvement. No gain would not prove the encoder or model family has
reached a ceiling. The task is unsuitable as an ungrounded question-only product
proxy because this candidate intentionally adds the QASC gold supporting facts.

## External paths

- Source TRAIN parquet: `C:\CodexArtifacts\pretrained-reuse-r2\qasc-evidence-context-20261104-attempt-01\source-train.parquet`.
- Enriched TRAIN rows and output will be created under the same new attempt root.
  No media, checkpoints, or feature arrays will be added to Git.
