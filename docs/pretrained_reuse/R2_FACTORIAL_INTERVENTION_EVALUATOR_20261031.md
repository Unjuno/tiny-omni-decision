# Paired factorial intervention evaluator — 2026-10-31

## Implemented

`src/tiny_omni_decision/factorial_interventions.py` scores already-saved
probability outputs from matched, missing, replaced, shuffled, time-offset,
contradictory, and option-permutation arms. Before scoring it requires unique
sample IDs, at least two nonempty source groups, fixed option cardinality,
identical per-arm option IDs/order, valid integer targets, and normalized
finite probability matrices. It writes ordered hashes for sample IDs, groups,
option order, and targets. Each arm reports Accuracy, NLL, Brier, ECE-15, and
mean confidence; intervention deltas versus the matched arm use paired
source-group bootstrap intervals for Accuracy, NLL, and Brier.

Sealed/test scoring requires `candidate_frozen=True` and a lowercase
64-character candidate SHA-256. This is a caller-supplied provenance gate, not
a cryptographic proof that the checkpoint or split is truly frozen; the helper
does not load media, models, or sealed data. Development evaluation remains
selection-affected. Interventions themselves must be generated upstream from
the predeclared task protocol; this helper cannot establish causal necessity,
valid pairing, participant independence, or rights.

## Verification

- `python -m pytest -q tests/test_factorial_interventions.py tests/test_fusion_diagnostics.py tests/test_cluster_comparison.py` — **17 passed**.
- `python -m ruff check src/tiny_omni_decision/factorial_interventions.py tests/test_factorial_interventions.py` — passed.
- `git diff --check` — passed.

No model, media, training run, validation cohort, or sealed audit was loaded.
This closes a reusable saved-prediction scoring/integrity component only; M-2
still lacks an admitted real task and does not authorize fusion fitting.
