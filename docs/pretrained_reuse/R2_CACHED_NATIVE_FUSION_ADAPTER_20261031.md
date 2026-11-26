# Cached native fusion adapter (2026-10-31)

## Change

Added `decide_cached_native_fusion`, a thin CPU/GPU-independent inference adapter
that connects the existing SHA-256-verified, causally aligned feature-cache
reader to `FusionDecisionScorer`. It validates the configured source slots,
shared clock metadata, typed option schema or supplied numeric option vectors,
and optional per-modality feature normalization. It abstains before calling the
scorer when any explicitly required modality has no fresh valid feature and
returns probabilities tied to the caller's option IDs otherwise. Free-text and
spoken criteria are rejected; typed criterion vectors must be explicitly
provided.

## Verification

- Regression test proves missing required video prevents scorer invocation,
  while fresh audio + video produce normalized option-ID probabilities.
- `pytest -q tests/test_native_feature_decision.py tests/test_fusion_decision.py tests/test_aligned_features.py`: **24 passed**.
- Ruff on the changed source and test: **passed**.

## Scope and limitations

This closes an inference/API integration gap only. It does not train or evaluate
a fusion model, validate raw-media encoders, prove that both streams are useful,
or satisfy R2's independent multimodal-quality gate. Cached feature timestamps
remain caller-supplied availability metadata; shared `clock_id` equality is not
a clock calibration proof. No media, model weights, GPU, or sealed audit were
used.
