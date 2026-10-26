# DROID temporal train-only comparison plan and protocol

**Goal:** Determine whether an invertible temporal representation changes
generalization within the 275 already frozen training recording groups.

**Architecture:** Two deterministic 112-value representations of the same 8x14
sensor window: original flattening, and final state followed by seven consecutive
coordinate increments. Both use the same 791-parameter seven-class linear head.
No command inputs, additional observation times, fusion, encoders or GPU.

**Tech stack:** Existing VideoPrism CPU Python environment with NumPy/SciPy;
reuse the exact original readout implementation from commit
`eba818067c6a9ef9089288205c26f6a8254ca6e5`. No upgrades.

## Frozen procedure before execution

- Pin the original train, sensor, label and extraction-summary hashes. Memory-map
  only the first 275 rows, whose train-first order is established by the pinned
  extractor and original train manifest; check all train labels.
- Five outer stratified recording-group folds, seed 17. Within each outer
  training set choose L2 among `[100, 1000, 10000]` using four grouped folds,
  seed 17 and macro NLL, ties prefer larger L2. Standardization and fitting use
  only inner training rows. Maximum 2000 optimizer iterations, fail on
  nonconvergence. Both arms share identical outer and inner fold identities.
- Fit each selected head on that outer training set and save probabilities on
  its untouched outer fold. Every train recording receives exactly one outer
  prediction per arm. Primary comparison: pooled outer macro NLL. Also save
  Accuracy, macro Accuracy, NLL, Brier, ECE-10 and per-class counts/metrics,
  fold-specific metrics and all tried inner-CV scores.
- No adaptive grid extension, representation adjustment, extra seed or validation
  evaluation follows the observed outcome in this experiment. No final product
  head is selected. This is development evidence on an already inspected train
  cohort; it is not blind quality evidence. Five outer folds are not five
  independent replications. No formal confidence interval is claimed.
- The transform is invertible and uses raw coordinate increments, without
  physical unit claims, angle unwrapping or division by varying time gaps.
  Changes mainly test standardization/regularization bias, not extra information
  or recurrent capacity. The two arms have equal parameter counts.

## Implementation and verification

1. Add failing tests for invertibility, order, finite/shape validation and
   outer-fold isolation; implement a small reusable comparison script.
2. Execute in a fresh output directory with a protocol saved before fitting.
   Preserve pinned source, fold assignments, probabilities, models and hashes
   outside Git. Never load validation feature rows or sealed audit.
3. Report the actual result, limits and warranted next action. Run Ruff, CPU
   tests and manifest validation, commit/push small artifacts and check CI.

User has delegated experimental strategy and opted to proceed autonomously.
This protocol implements the previous train diagnostic's next action.
