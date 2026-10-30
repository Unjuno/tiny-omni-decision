# Text-free native-feature Decision readout — 2026-10-31

## Implemented

`src/tiny_omni_decision/native_feature_decision.py` adds a small shared readout
for one frozen modality feature and an explicit candidate feature matrix. The
readout has no tokenizer, model factory, text encoder, prompt, or option-name
input. `TypedOptionSchema` encodes explicitly declared numeric attributes with
predeclared center/scale and enum attributes with a fixed one-hot vocabulary;
unknown fields, categories, and free-form strings are rejected. Option IDs are
only carried into the output JSON. The model scores each candidate independently
against the same observation representation, so reordering candidates reorders
the logits exactly.

The `decide_cached_native_feature` adapter validates and reads one selected
modality feature stream through the existing cache contract, checks SHA-256 and
metadata alignment, applies the query-time freshness gate, and abstains when no
valid feature is available. It can encode the request's typed options using the
fixed schema above; it rejects text criteria and mixed-modality requests.
Raw-media encoders are still outside this module; features must already exist in
the immutable hashed cache.

## Verification

- `python -m pytest -q tests/test_native_feature_decision.py tests/test_aligned_features.py tests/test_native_observations.py tests/test_fusion_decision.py` — 38 passed.
- `python -m ruff check src/tiny_omni_decision/native_feature_decision.py tests/test_native_feature_decision.py` — passed after correcting the optional-Torch import ordering.
- Tests cover candidate permutation equivariance, caller option-order preservation,
  normalized probabilities, padded invalid candidates, nonfinite-value rejection,
  invalid candidate sets, cache hash rejection, freshness/missing-evidence
  abstention, fixed-schema numeric/enum option encoding, and rejection of text
  criteria or free-form option attributes.

No model was trained, no feature cache or evaluation set was loaded, and no quality
or parameter-budget claim is made by this implementation test.

## Remaining integration gate

Define one task-specific typed schema and complete a train/validation experiment
before counting this as an M1 Decision result. The general numeric/enum encoder
does not decide which attributes are valid task evidence; prevent target, source,
filename, and sampling metadata from entering the schema.
