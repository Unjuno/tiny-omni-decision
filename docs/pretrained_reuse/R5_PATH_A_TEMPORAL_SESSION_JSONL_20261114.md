# Path A JSONL temporal feature sessions — 2026-11-14

## Purpose and boundary

This increment exposes the existing bounded `SessionTemporalFeatureState` through the canonical Path A JSONL process. It supports caller-supplied frozen feature updates, explicit missing updates, causal snapshots at an `as_of_ms` time, and session close. It does not encode media, align clocks, score options, train a model, or establish temporal or multimodal quality.

This is R5 runtime plumbing only. The snapshot returns feature payloads so a separate, identity-checking consumer can reuse them. It is **not connected to a trained fusion scorer** in the canonical four-route runtime; snapshot output is not a Decision result.

## JSONL operations

All requests and responses use one JSON object per line. Existing `task: text|image|audio|video` requests retain their prior behavior. Temporal operations use these exact top-level operation names:

- `observe`: append a feature event with `session_id`, `clock_id`, `modality`, `source_id`, `available_at_ms`, a complete `ObservationFeatureKey` object, and `feature_payload_base64`.
- `record_missing`: append an explicit missing event for one stream/time. A prior feature is not silently reused after that event.
- `snapshot`: provide `session_id`, the same `clock_id`, `as_of_ms`, `max_age_ms`, and a non-empty list of `{modality, source_id}` streams. Only the latest causal non-stale value is returned. Missing, future, stale, and evicted values have a reason and no payload.
- `close_session`: clear one process-local session and return its discarded event/byte counts.

Feature payloads are canonical little-endian float32 bytes encoded as base64. The feature key must declare `feature_dtype: "float32"`, matching shape, modality, and source ID. The wrapper validates key structure, byte length, finite values, per-payload encoding, stream identity, timestamps, and the payload's own SHA-256 when stored. The encoder and preprocessor hashes remain caller-provided claims; the runtime does not load those encoders or prove their provenance. A downstream scorer must compare returned feature keys against its own expected identities before using the values.

`runtime_sha256` is computed from the loaded assembly's `package-manifest.json`; callers cannot override it. Each session binds to its first `clock_id`; runtime or clock mismatches fail closed. Timestamps and common-clock truth remain the caller's responsibility. State is in memory only and disappears when the process exits.

## Resource bounds

- At most 8 open sessions per process.
- At most 1 MiB of serialized state per session (up to 8 MiB across sessions); Python object overhead is excluded from this accounting.
- At most 16 streams and 8 events per stream. Evictions are returned in operation status and unavailable snapshots identify evicted history.
- Close a session explicitly before opening another after the session limit is reached.

These are state-contract bounds, not measured process RSS or deployment limits.

## Verification

Source branch before the change: `codex/path-a-canonical-json-runtime-20261114` at `d67212eb4dc2d615c502b4142ade0694122b0b45`.

The new CPU-only tests use synthetic, hash-pinned float32 vectors and the real session-state implementation. They cover: causal payload reuse; stale and explicit-missing behavior without leaking prior payloads; clock mismatch and unknown-session rejection; bounded session admission and close; invalid-input rollback; and visible bounded-history eviction. Focused tests pass: 42 across the new session tests and the existing JSONL, temporal-state, and native-feature decision tests. Ruff passes on the changed implementation and tests.

No model weights, feature cache, training run, dataset, or sealed evaluation was opened or changed. No GPU was used. There are no Accuracy, NLL, Brier, ECE, latency, or resource-quality claims from this change. The existing model assembly files were not changed; their manifest records 170,499,716 unique parameters and 687,254,772 bytes. The code change adds no model weights or parameters. The installed software bundle size was not remeasured.

## Remaining gates

This API makes time-indexed frozen features reusable within a live process; it does not show that the Path A scorer benefits from them. The next quality-bearing step still requires an admitted task with independent groups, fixed temporal and missing-input controls, and a scorer whose expected feature identities are explicit. TartanAviation ATC labels do not pass that quality gate: only 192 positive human-reviewed rows across 191 clips were observed, with one provided train split and no speaker IDs. Do not fit a joint model or claim temporal/fusion improvement from this interface work.
