# R5 temporal state concurrency and isolation contract — 2026-11-12

## Scope

Added CPU-only contract coverage for `SessionTemporalFeatureState` in
`tests/test_temporal_feature_state.py`. The test uses synthetic feature bytes;
it does not load a model, media, validation corpus, or sealed audit. No GPU or
training was used.

## Evidence

- Source commit before this change: `d45e4570c1898a77419182e6bf1509df59302002`
- Isolated branch: `codex/r5-temporal-session-isolation-20261112`
- A barrier-coordinated, 64-iteration test runs separate audio and video stream
  updates alongside snapshots. Every returned feature is checked to be
  available no later than the query's `as_of` time, with a matching payload
  digest and stream identity. The same run exercises FIFO eviction and checks
  event/serialized-byte limits.
- A separate test creates two state objects with the same runtime, clock,
  modality, and source IDs but different payloads, then confirms each snapshot
  returns only its own session's payload.
- Existing tests already reject duplicate and decreasing timestamps for one
  stream; this change retains that coverage.
- Initial focused run: `python -m pytest tests/test_temporal_feature_state.py
  -q` — **14 passed**.
- Related CPU regression run: `python -m pytest
  tests/test_temporal_feature_state.py tests/test_temporal_native_decision.py
  tests/test_observation_alignment.py tests/test_feature_cache.py -q` —
  **47 passed**.
- Concurrency stability check: the concurrent-stream test passed **8/8**
  separate pytest process runs.
- `ruff check tests/test_temporal_feature_state.py` — passed.

This verifies the in-memory state-store contract for concurrent calls and
instance isolation. It does not establish concurrent scorer/model-worker
behavior, correct real-world clock synchronization, persistence across process
restarts, or any decision-quality improvement. No production code changed
because the tested contract passed against the existing implementation.
The full repository suite was not run because unrelated tests may initialize
GPU runtimes while the separately owned Orbitune GPU job is active.
