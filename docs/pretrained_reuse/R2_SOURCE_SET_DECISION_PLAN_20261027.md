# Set-valued source decision prototype

Goal: preserve the existing candidate scorer and test a separate factorized
visibility-plus-membership objective with explicit none-visible exclusivity.
The baseline softmax and old artifacts are read-only. No claim of calibrated
probabilities follows from the parameterization.

1. Add a torch-only helper `scripts/source_set_decision.py` operating on the
   existing padded `[batch, candidates + none]` logits and boolean targets.
   Use binary cross entropy for the none event, plus per-record mean candidate
   binary cross entropy only on visible records. Mask all padded candidates;
   reject invalid targets and visibility combinations.
2. Decode with a fixed none threshold 0.5, followed by candidate conditional
   sigmoid threshold 0.5 when visible. If none pass, return all candidates
   tied for the largest valid score; never include none and a visible candidate together. This is an
   exploratory point-decision rule, not a coherent joint distribution.
3. Add tests for multi-target membership, padding invariance, none exclusivity,
   fallback, gradient flow, and rejected inconsistent labels. Run targeted
   tests and lint. Preserve the old readout, protocol and outputs.
4. Before any train run, fix a separate protocol and compare the same frozen
   features, source splits, seeds, parameter count, and selection budgets.
   Measure exact set, per-option proper scores, none-event metrics, unimodal
   controls and calibration on truly independent examples. Existing observed
   folds are exploratory; do not use sealed audit for tuning. Save checkpoints
   and receipts only when adequate storage is available.
