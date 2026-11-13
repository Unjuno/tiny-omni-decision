# Bounded temporal feature-state runtime contract — 2026-11-03

## Scope

This change implements a CPU-only, in-process session buffer for reusing
caller-provided frozen observation features across delayed Decision requests.
It is a runtime correctness contract, not a learned world model, a quality
result, or a persistence-on-disk implementation. It does not read media, align
clocks, invoke encoders, or modify model weights.

## Contract

`SessionTemporalFeatureState` accepts feature events keyed by modality and
source. Every feature retains its complete `ObservationFeatureKey` identity,
including source revision, raw observation hash, encoder weights, preprocessor,
processing hash, dtype, shape, and temporal policy. The session is also bound to
one caller-supplied runtime SHA-256 and one already-verified clock ID. A change
requires explicit `rebind`, which clears the old events before accepting the
new runtime/clock.

Evidence timestamps mean the time the feature became available, on the shared
clock established by the caller. Per-source updates must be strictly
increasing. A snapshot at `as_of_ms` returns only features at or before that
time and within the caller's maximum age. An explicit missing event masks
older values; future, stale, unseen, and evicted history are surfaced as
unavailable states instead of reusing stale evidence.

State is bounded by serialized event bytes, events per source, and source count.
Byte accounting includes canonical JSON metadata plus feature payload bytes,
but excludes Python object/container overhead. Deterministic FIFO eviction is
counted. An individual event larger than the entire budget is rejected without
mutating existing state.

## Synthetic CPU verification

Tests use only synthetic typed keys and byte payloads. They cover delayed
queries over repeated video/audio features, explicit missingness, future and
stale evidence, strict per-source ordering, runtime/clock mismatch and explicit
invalidation, feature identity, FIFO/event/stream/byte limits, and no mutation
on rejected oversized writes.

Commands:

```text
ruff check src/tiny_omni_decision/temporal_feature_state.py tests/test_temporal_feature_state.py
python -m pytest -q tests/test_temporal_feature_state.py tests/test_observation_alignment.py tests/test_feature_cache.py
```

Result: Ruff passed; 40 focused tests passed. No pretrained checkpoint,
dataset, media, validation IDs, sealed audit, GPU, or cloud compute was used.

## Limits and next gate

The state buffer is not wired into `PathARuntimeDispatcher` or a trained
readout, and its synthetic tests establish no decision quality or H4 memory
benefit. It has no disk serialization/restart contract. Before integration, a
source/task and a task-specific delayed-query/missing-observation protocol must
be admitted and fixed; then compare with a no-memory baseline on independent
groups. R2-M, the four-modality quality target, deployment parameter/package
limits, and product qualification remain open.
