# R2 text-criterion to typed-option contract (2026-11-05)

## Result

`decide_cached_native_fusion` now accepts a non-typed criterion when the caller
provides its precomputed criterion feature vector. For text criteria, the
caller may instead pass a frozen encoder callback. It still consumes only
hash-checked cached modality features; it imports or loads no text or speech
encoder and reads no raw media. A criterion present in the request without
either a feature vector or the supported text callback fails closed.

A CPU-only synthetic test uses deterministic hand-set scorer weights to show
that changing only the criterion feature polarity changes the selected typed
option, that reversing supplied option order preserves probabilities by
option ID, and that save/reload reproduces the exact response. The criterion
vectors are supplied directly by the test; no learned text encoder or model
quality is being claimed.

The existing text-free `decide_cached_native_feature` interface continues to
reject a criterion. The new contract is specific to the feature-fusion scorer.

## Verification

- `python -m pytest tests/test_native_feature_decision.py -q`: 16 passed.
- `ruff check src/tiny_omni_decision/native_feature_decision.py tests/test_native_feature_decision.py`: passed.
- Whole-file `ruff format --check` reports formatting differences in both
  touched files, including pre-existing sections; no broad formatting rewrite
  is included in this narrow change.

## Limits and next gate

This is an interface contract only. A text/spoken criterion encoder had not
yet been connected when the first result above was recorded. Real
text-conditioned option selection, task-admitted data, generalization,
calibration, latency and multi-input quality remain unverified. R2-M1/R2-M2
stay open; this result does not justify training or model promotion.

## Follow-up: pinned MiniLM criterion feature path (2026-11-05)

`TypedTextDecisionRuntime.encode_criterion()` now exposes one criterion vector
from the same frozen MiniLM-L3 encoder already used by the Path A text runtime.
It validates nonempty input and returns a finite `[384]` pooled feature. Its
method can be passed directly as the cached native fusion `criterion_encoder`
callback. The model is the existing
`sentence-transformers/paraphrase-MiniLM-L3-v2` revision
`4ca70771034acceecb2e72475f72050fcdde4ddc`, with encoder weights SHA-256
`cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.

An actual CPU-only load from the preserved Path A runtime bundle produced two
finite 384-wide vectors for “higher score” and “lower score” criteria. Their
cosine similarity was `0.84799665`. This establishes only that the encoder
produces different features for the two strings; it does not show the fusion
scorer understands either instruction. No labels, task corpus, validation,
sealed audit, GPU, training or checkpoint selection were used. The bundle and
its weights were read-only.

- `python -m pytest tests/test_text_runtime.py tests/test_native_feature_decision.py -q`: 26 passed.
- `ruff check src/tiny_omni_decision/text_runtime.py tests/test_text_runtime.py`: passed.
- `git diff --check`: passed.

The end-to-end CPU smoke then passed the real MiniLM criterion callback,
hash-checked synthetic sensor cache and typed candidates through the fusion
JSON endpoint. It produced normalized two-option JSON for both criteria and
matched the exact first response after saving/reloading the randomly initialized
fusion scorer. The two selected options happened to differ, but the random
scorer is untrained and this must not be interpreted as criterion-following
quality. No labels or quality metrics were used.
