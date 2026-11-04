# QASC text-criterion FusionDecisionScorer protocol (2026-11-05)

## Question

Can the newly connected text-state / natural-language-criterion / supplied-
option `FusionDecisionScorer` learn this one fixed text-only Decision task
under the same five grouped QASC TRAIN folds used by the existing Path A
readout?

This is a single fixed CPU cross-validation run, not a model or hyperparameter
sweep. The intended factor is the input-role decomposition: state is the two
publisher-provided support facts, criterion is the question, and each supplied
choice has its own frozen feature. The model is trained from a fresh readout
initialization independently in each fold. The pretrained MiniLM remains
frozen.

## Frozen inputs and procedure

- Use only the 8,134-row QASC `train` evidence corpus at
  `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`, CC BY 4.0.
- Corpus SHA-256:
  `6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b`.
- Reuse the existing five grouped folds without modification. The fixed
  question-only Path A `fold-assignment.jsonl` SHA-256 is
  `72282c4b57fb2e5a56c4d2f15f75d11bf11f79071d8ac8a981b1a7311bcd0b67`.
  Rebuild and audit the grouping from TRAIN rows, then require exact row-ID to
  fold parity before fitting.
- Frozen MiniLM-L3 revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`, weight SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Masked-mean pool state text (`fact1`, then `fact2`), question criterion, and
  each candidate separately, using the existing 256-token truncation contract.
- Train `FusionDecisionScorer({"text_state":384}, option_dim=384,
  criterion_dim=384, width=128)` with 8-way cross entropy, seed 17 per fold,
  AdamW `lr=1e-3`, weight decay `1e-4`, batch size 64, and exactly 8 epochs.
  Use the fixed epoch-8 endpoint; do not select a checkpoint from its held fold.
- Report OOF Accuracy, NLL, multiclass Brier, ECE-15 and mean confidence,
  per-fold scores, uniform and fold-training position-prior controls, and a
  same-checkpoint criterion-masked ablation. Check option permutation and
  exact save/reload parity.
- Store predictions and fold checkpoints under the new external directory
  `C:\CodexArtifacts\pretrained-reuse-r2\qasc-text-criterion-fusion-cv-20261105-attempt-01`.
  Refuse to overwrite it.

## Integrity and claim limits

The runner must not open QASC validation/test or sealed-audit inputs. QASC is a
text-only task, so this cannot establish audio, image, video, sensor or joint
multimodal quality. The support facts contain the normalized target option in
6,167/8,134 rows (75.82%); this creates a strong answer-span shortcut risk.
The criterion-masked ablation helps diagnose criterion use but does not replace
a same-state reversed-criterion task. Existing QASC CV uses the same TRAIN
records, so this is a different readout formulation on already-used TRAIN
data, not independent dataset evidence. It must not be used to reopen any
exposed validation/test or tune other experiments.

No quality threshold or model-promotion rule is relaxed. A poor result is
retained. No additional seed, training budget, or architecture variant is
authorized by this protocol.
