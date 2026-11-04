# QASC TRAIN-only grouped cross-validation protocol — 2026-11-04

## Purpose

Measure the existing frozen-feature text Decision readout under an internally
held-out, source-grouped design. This is one narrow text-only development task,
not broad reasoning quality. It does not reuse the exposed QASC validation shard
or open the official test shard.

## Fixed conditions

- Source: `allenai/qasc@c44a320c2bb682d614a675d2ba4be072150db4eb`, prepared mirror
  `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, CC BY 4.0.
- Rows: exact prepared QASC TRAIN JSONL SHA-256
  `d2e220510f98e19a2df5e44f89d29e1b1bd5b218397264b08ff1102ea3903e51` (8,134 rows).
- Frozen text features: MiniLM-L3
  `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`;
  read-only cache SHA-256 `997029c92fe486a2279595da994847a2c0d5442649e667cbf5e4db6b56fcad16`.
- Readout: existing 196,865-parameter `FrozenFeatureCandidateScorer`, input
  `[question, option, |difference|, product]`, hidden size 128. The MiniLM is
  frozen; only the scorer is trained.
- Seed 17 for every fold, five deterministic grouped folds, eight fixed epochs,
  AdamW `lr=1e-3`, `weight_decay=1e-4`, batch 64 questions, no option-order
  augmentation. Every fold uses the epoch-8 endpoint; no held-fold checkpoint
  selection or hyperparameter tuning.
- CPU only. Existing frozen cache is read for TRAIN features/targets only.
  Validation and test feature/label members are not indexed by the runner.

## Leakage gates

Rows are connected when they share any of: unordered normalized fact-pair group,
NFKC/casefold normalized question, or normalized question+sorted options.
Entire connected components are assigned to exactly one fold. Before fitting,
all five held-fold comparisons must show zero overlap for all three identities.
All 8,134 examples remain in exactly one held fold; no row is removed.

## Metrics and artifacts

The primary metric is pooled out-of-fold Accuracy, with NLL, multiclass Brier,
15-bin ECE and mean confidence. Report each fold separately. Compare uniform
choice and a training-fold answer-position prior. Store all out-of-fold IDs,
targets, predictions and eight probabilities; save/reload each fold's epoch-8
scorer and require exact probability equality.

Use a new external output directory under
`C:\CodexArtifacts\pretrained-reuse-r2\qasc-path-a-train-only-group-cv-20261104-attempt-01`.
Do not put weights or frozen features in Git. A dry-run produces the split audit
before any fit. No change to the existing QASC run, validation, or test data.
