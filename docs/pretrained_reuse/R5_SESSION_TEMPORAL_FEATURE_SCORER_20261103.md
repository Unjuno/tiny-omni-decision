# R5 session temporal feature-to-Decision integration — 2026-11-03

## Change

Added `decide_temporal_native_fusion`, a local in-process adapter from `SessionTemporalFeatureState` to the existing `FusionDecisionScorer` and typed option JSON output. It selects the latest evidence available as of the query time, enforces a maximum age, verifies the expected per-stream encoder/preprocessor identity and each payload SHA-256 while allowing input hashes/timestamps to change between events, decodes only canonical little-endian float32 vectors, and abstains before model inference if any required modality is missing or stale. Options remain typed numeric/enum fields; no text or media encoder is loaded by the adapter.

The session state remains caller-owned and process-local. The caller must supply features, a truthful common clock, and features whose encoder receptive fields are causal for their declared availability times. The adapter does not align clocks, read raw media, persist/recover state across restarts, update model parameters, or establish model quality.

## CPU integration check

A synthetic two-stream fixture inserted three feature events (two video updates and one audio update) and asked two delayed questions at different `as_of_ms` times. The real scorer received the earlier video feature for the first query and the later video feature for the second; audio was reused, and the state event count remained three. A second case made both required streams unavailable/stale and verified abstention occurred before scorer invocation. A third case verified that a feature key from another encoder fails closed.

Verification on the current branch:

- Full CPU test suite: **2,806 passed, 11 skipped**.
- Focused state/integration/scorer tests on the final implementation: **29 passed**.
- Ruff: **all checks passed**.
- `git diff --check`: passed.
- GPU training/inference, source media, checkpoint loading, and sealed-audit access: **not performed**.

These are contract/runtime results on synthetic features only. They add no task quality, calibration, synchronization, or real-device performance evidence and do not close R2-M. R5 still needs a rights-cleared real stream and task-specific repeated-query evaluation before the state mechanism can be selected for product use.
