# QASC TRAIN-only Grouped Cross-validation Result — 2026-11-04

## Scope and execution

This was a CPU-only text control, not a multimodal evaluation. It used the pinned,
CC BY 4.0 prepared QASC TRAIN rows and the existing frozen MiniLM-L3 option-feature
cache. The encoder was not trained; the 196,865-parameter option scorer was fit
independently in five folds. Each fold used seed 17, eight fixed epochs, AdamW
(learning rate 1e-3, weight decay 1e-4), batch size 64 questions, and the epoch-8
endpoint. No held-fold checkpoint selection or hyperparameter search was done.

The first dry-run found a fold-balancing bug (fold sizes 9, 3,241, 3,240, 10,
1,634). Training did not start. The assignment objective was corrected to score
the global fold balance, then re-audited. Final held-fold sizes are 1,626–1,627.
All five folds have zero cross-fold overlap for fact-pair group, normalized
question, and normalized question+choice content. Every one of 8,134 TRAIN rows
is held out exactly once. The official QASC validation shard, official test shard,
and sealed audit were not opened or scored. The preserved feature-cache container
also includes historical development entries; the runner indexed only its
`train_features` and `train_targets` members. No development labels were used.

The prepared TRAIN JSONL hash is
`d2e220510f98e19a2df5e44f89d29e1b1bd5b218397264b08ff1102ea3903e51`; the frozen
feature cache hash is
`997029c92fe486a2279595da994847a2c0d5442649e667cbf5e4db6b56fcad16`.
There are 7,139 fact-pair groups. Total training presentations across folds were
260,288 (32 epochs over the corpus in aggregate) and 4,080 optimizer updates.
The readout plus frozen encoder contains 17,586,689 parameters. All five fold
checkpoints were saved and reloaded; probability arrays matched exactly.

## Out-of-fold results

| Evaluation | n | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Grouped 5-fold OOF readout | 8,134 | 0.3535 | 1.9197 | 0.8112 | 0.1727 | 0.5262 |
| Uniform 8-choice baseline | 8,134 | 0.1242 | 2.0794 | 0.8750 | 0.0008 | 0.1250 |
| Fold-training answer-position prior | 8,134 | 0.1290 | 2.0791 | 0.8749 | ~0.0000 | 0.1290 |

| Fold | Held rows | Accuracy | NLL | Brier | ECE-15 | Epoch-8 online train CE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,627 | 0.3307 | 1.9670 | 0.8376 | 0.1916 | 0.8549 |
| 1 | 1,627 | 0.3479 | 1.9537 | 0.8167 | 0.1810 | 0.8402 |
| 2 | 1,626 | 0.3616 | 1.8938 | 0.8006 | 0.1647 | 0.8563 |
| 3 | 1,627 | 0.3688 | 1.9077 | 0.8025 | 0.1611 | 0.8446 |
| 4 | 1,627 | 0.3583 | 1.8765 | 0.7986 | 0.1653 | 0.8473 |

The readout exceeds both simple baselines on this internal, grouped QASC TRAIN
cross-validation. Its mean confidence (0.526) is high relative to its accuracy
(0.353), with ECE-15 0.173; calibration remains poor. Online train CE is a rolling
training measure, not a fixed-example generalization gap. The results support
that the cached language features contain useful signal for this narrow eight-way
science-question task, while showing substantial residual error and overconfidence.
They do not establish generalized text decision quality, multimodal fusion,
95–98% accuracy, or product readiness. Do not use this result to promote a model
or to reopen the exposed validation/test shards.

## Runtime and artifacts

- Source commit: `a8c1358c9ca132de73b6bda7074432d384654c12`.
- Python 3.11.9; PyTorch 2.6.0+cu124; NumPy 2.4.6; Transformers 5.6.2.
- CPU wall time: 32.79 s; measured fit time 30.35 s; evaluation/reload 0.58 s.
- Training compute cost: none; GPU was not used.
- New output directory: `C:\CodexArtifacts\pretrained-reuse-r2\qasc-path-a-train-only-group-cv-20261104-attempt-01`.
- Main report SHA-256: `3dca01b0ede00158d304ea646f869bbc8178f9cf46f7e3b8705efb2487e5e783`.
- OOF predictions SHA-256: `b32782c8dbd4343f46036eaff14aee4910f69d84e5180fdca8b066fc9987db1c`.
- Effective config SHA-256: `47e211e67887243b4484ac9be869043f1158700a938056863ea90ec8226bba0d`.
- Fold plan SHA-256: `bf6f9fb6653484915313dc5e908b226b333ebd49254546020e08a4d382f249d4`.
- Fold assignment SHA-256: `72282c4b57fb2e5a56c4d2f15f75d11bf11f79071d8ac8a981b1a7311bcd0b67`.

The external directory contains each fold checkpoint, effective config, fold plan,
fold assignment, per-row OOF probabilities, aggregate report, and answer-position
baseline summary. Checkpoints and source data remain outside Git.

## Decision

This completes one limited train-only text diagnostic. Keep its readout unpromoted.
It does not close R2-M's requirement for a real, task-admitted multi-input task
where both modalities are necessary, nor the broader per-modality quality gates.
