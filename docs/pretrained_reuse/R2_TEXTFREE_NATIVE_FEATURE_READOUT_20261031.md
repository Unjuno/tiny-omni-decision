# Text-free native-feature Decision readout — 2026-10-31

## Implemented

`src/tiny_omni_decision/native_feature_decision.py` adds a small shared readout
for one frozen modality feature and an explicit candidate feature matrix. The
readout has no tokenizer, model factory, text encoder, prompt, or option-name
input. A task-owned adapter must encode candidate attributes from a fixed typed
schema (for example numeric action fields or enum codes); option IDs are only
carried into the output JSON. The model scores each candidate independently
against the same observation representation, so reordering candidates reorders
the logits exactly.

The module deliberately does not load raw media or feature caches. Callers must
obtain and validate the selected modality feature through the existing
feature-cache/encoder path, and must gate stale or missing evidence before
calling the scorer. This is a reusable CPU-testable readout primitive, not the
complete cache-to-Decision runner requested by R2-M.

## Verification

- `python -m pytest -q tests/test_native_feature_decision.py tests/test_native_observations.py tests/test_fusion_decision.py` — 27 passed.
- `python -m ruff check src/tiny_omni_decision/native_feature_decision.py tests/test_native_feature_decision.py` — passed after correcting the optional-Torch import ordering.
- Tests cover candidate permutation equivariance, caller option-order preservation,
  normalized probabilities, padded invalid candidates, nonfinite-value rejection,
  and invalid candidate sets.

No model was trained, no feature cache or evaluation set was loaded, and no quality
or parameter-budget claim is made by this implementation test.

## Remaining integration gate

Connect this scorer to an immutable single-modality feature-cache adapter and a
task-specific deterministic typed-option encoder. Then verify input hash/source
identity, option permutation, missing/stale-evidence abstention, and one complete
train/validation experiment before counting it as an M1 Decision result. Do not
pass text descriptions through an implicit embedding model.
