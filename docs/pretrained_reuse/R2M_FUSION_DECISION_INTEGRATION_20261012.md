# Masked fusion to option probabilities: implementation evidence

## Scope and design

The previous masked vector mixer had no candidate/criterion interface. The new
`FusionDecisionScorer` composes it with the existing shared MLP candidate scorer.
It accepts named feature vectors, explicit modality availability, typed or
externally encoded option vectors, and an optional criterion with its own
presence mask. It has no option positions, candidate-specific output weights,
mandatory text encoder, or generative JSON decoder. Invalid padded candidates
receive zero probability; each row must contain at least two valid options.

Presence bits are separate from feature values. Missing inputs and padded
options are sanitized before learned operations, including NaN/Inf placeholders.
All-modality absence is rejected. Task-specific required sources and upstream
time/unit/payload checks still belong to the existing evidence gate. Having one
available input does **not** establish enough evidence for a multi-input task.

This is a vector control, not a temporally indexed state model. It does not yet
connect all pretrained towers, understand arbitrary descriptions, calibrate
confidence, or satisfy the complete R2-M matrix. Per-modality source projections
are followed by a shared mixer and scorer; no independent task-head ensemble is
introduced. Weighted pooling may discard useful source-specific information and
must be compared against token-preserving alternatives on real tasks.

## Tests and actual smoke

Four new tests cover candidate permutation/padding, optional criterion and absent
feature isolation, criterion gradient arrival, all-source absence, insufficient
options and nonfinite available options. Three existing mixer tests cover missing
feature gradients and valid zero input. The new tests first failed because the
module did not exist, then passed after implementation.

Command (separate process; local CPU):

```text
python -m scripts.smoke_fusion_decision --output C:/CodexArtifacts/pretrained-reuse-r2/fusion-decision-truth-smoke-01
```

Predeclared smoke: seed17, 512 AdamW updates, lr0.01, weight decay0.01, width16,
eight exhaustive Boolean triples. Two input bits and a criterion bit define
three-way parity. All eight rows are training fixtures; there is **no held-out
test**, and no generalization or multimodal quality percentage is inferred.
No real media, pretrained weights, sealed audit, GPU or paid compute was used.

Measured: 466 trainable parameters, 0.822 seconds for training/checks/save,
8/8 training fixtures correct. Flipping either input or the criterion yields
the corresponding changed target in 8/8 fixtures for each intervention.
These interventions also stay within the training truth table. Candidate-order
permutation and fresh checkpoint reload reproduce probabilities exactly. All
fixture outputs were serialized through the existing deterministic JSON adapter.

The artifact directory contains config with exact executed source hashes,
all512 loss/gradient records, final weights, probabilities and JSON results.
Final weight SHA-256:
`794e8770dcfb69fe06e669fa308b622afd51000b564ff207b6805d8c5da18e6c`.
Parent source commit was188ec30; config hashes identify the uncommitted code
actually executed. Subsequent runner edit only wrapped a long source line for
lint. No checkpoint selection or extra seeds were used.

## Remaining quality evidence

This closes a missing implementation connection, not a quality gate. Real
paired-task training, calibration, candidate-semantic transfer, temporal
alignment, complete resident-parameter accounting and per-modality/joint
held-out quality remain required. The negative VIDIMU and STARSS23 results
remain unchanged and are not replaced by this deterministic logic smoke.
