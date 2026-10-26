# Verified native feature handoff: executed CPU replay

The new `decide_bound_native` entry point binds the shared text-free JSON
boundary to current local media, a full encoder/preprocessing cache identity,
and a producer-pinned feature payload. A real CPU replay passed on 24 existing
training requests across audio, image and video. This closes a specific input
integrity gap; it does not improve measured model quality or close R2 M-0/M-1.

## Implemented behavior and trust boundary

`NativeFeatureBinding` belongs to trusted application/extractor configuration,
separate from request JSON. It carries an `ObservationFeatureKey`, independently
pinned feature-payload SHA-256, source ID, clock and exact timestamps. A complete
request must match those fields. The loader hashes the actual local media on
every access, loads only that complete cache key, compares the cache payload
against the independent receipt, and checks float32 byte count, shape and finite
values before dispatching to the existing Audio/Image/Video readout. Shape or
checksum checks alone would not detect a validly serialized wrong cache.

Same-content path aliases work. Same-path/same-size/same-mtime replacement fails.
Changing encoder weights, preprocessing, clock or timestamps cannot silently
reuse the old cache. Indexed containers require an explicit index contract and
currently fail. Incomplete observations abstain without reading a feature or
inventing probabilities. Selected JSON retains the existing option IDs,
attributes, probability order, confidence and tie handling, adding only a
`feature_provenance` object. Existing tensor-level adapters remain low-level APIs
whose callers are responsible for tensor provenance.

This authenticates a handoff **relative to a trusted producer receipt**. Hashes
cannot establish that an untrusted party really executed its declared encoder.
The replay imports legacy feature rows from hash-pinned extraction containers;
it does not rerun encoders or create new extraction evidence. The import key
includes the legacy container hash, encoder identity and weights, import-code
hash and time metadata. Historical source/archive and extraction limitations
remain in force. Product configuration must pin producer contracts independently
of user requests and match them to the trained readout.

## Actual replay

`scripts/verify_native_feature_binding.py` verifies 13 legacy input hashes before
and after execution, imports the first eight train rows per modality into a
fresh immutable cache, and compares the verified JSON to the existing direct
feature JSON. It also reverses options and attempts wrong-feature-digest and
wrong-clock requests for every example. Selected source media are rehashed after
execution and complete readout states are checked unchanged.

| Path | Saved readout | Requests / distinct media | Direct vs bound JSON | Median bound decision |
|---|---|---:|---|---:|
| Audio | `textfree-audio-decision-03` | 8 / 8 | Exact | 1.331 ms |
| Image, color | `textfree-image-decision-512-01` | 8 / 8 | Exact | 1.485 ms |
| Video, typed program | `textfree-video-clevrer-typed-fit-01`, BiLSTM | 8 / 4 | Exact | 3.960 ms |

There are 20 distinct media assets and 20 feature entries (158,832 stored cache
bytes). The repeated video scenes reuse the same cache entries across different
typed programs. Option reversal preserves all selected IDs; the maximum
probability delta after identity alignment is `1.1920928955078125e-7`.
All 48 wrong-digest/clock controls are rejected. No original media, weights,
corpus, feature cache, or extraction receipt was changed.

The reported decision timer includes actual media hashing, cache verification,
tensor conversion and the small readout. It excludes encoding, source-cache
import, model load, process start and Python imports. It is not end-to-end raw
media latency or a comparison with an unverified path. Total measured replay
body time is 1.426 s, excluding Python/module startup. No encoder or GPU ran;
CUDA stayed uninitialized. The existing local CPU environment and four threads
were used. No cloud spend. Peak RAM was not instrumented for this replay.

Video PTS are reconstructed from the same preserved raw videos with the legacy
endpoint-inclusive frame-index policy; declared and decoded frame counts must
agree. These clocks are clip-local. No synchronization between unrelated audio
and video clips is asserted. Audio and still images use a single observation
timestamp at zero.

## Exact outputs and reproducibility

Authoritative output:
`C:/CodexArtifacts/pretrained-reuse-r2/native-feature-binding-replay-20261027-attempt-03/`.
The [small result receipt](R2_NATIVE_FEATURE_BINDING_20261027.json) retains each
probability JSON, provenance key, timing and negative-control outcome. Cache
payloads and full bindings remain outside Git.

- Result SHA-256: `9fed1e615e4dde72e455b185b7ef5b2ccfa3f4639db679e0b5e11140ccc63356`.
- Bindings SHA-256: `d9812d8105b8a5cf6b5e687d6c049c0d46922a62dae0a2fa3102fbfb3ea2902d`.
- Executed replay script SHA-256:
  `b27d5d5a536c38029c37b02edb885fff829ffe13ed6ed8852b7acfb4fa525259`.
- Executed module SHA-256:
  `3ba6d37b78ba5112b3186e3c983e91f56134b38f469b771ee46f6ff9b36156c7`.
- Base source commit: `d8a8f34fea0f0dd2e66c3bfc1b532d51de0f9065`;
  new code was uncommitted during execution and is bound by these hashes.

Attempts 01 and 02 are preserved. Attempt 01 established parity; 02 added
explicit negative controls and post-run input hashes; 03 also checked the
legacy video frame-count policy and post-run source media. No model, feature,
sample or selection policy changed. A command before attempt 01 failed at import
because `PYTHONPATH` pointed at the installed legacy worktree; it created no
run output. The explicit repository source path below avoids that ambiguity.

```powershell
$env:PYTHONPATH='src;.'
python -m scripts.verify_native_feature_binding --output C:/CodexArtifacts/pretrained-reuse-r2/native-feature-binding-replay-NEW
```

No labels are used to score quality. No validation set is selected or scored;
the legacy serialized audio feature file necessarily includes its old validation
arrays, but only train rows are selected for replay. No test or sealed audit is
opened. The poor or narrow historical quality results remain unchanged.

CPU tests cover stale actual media, aliases, altered full keys, validly
serialized wrong feature payloads, invalid payload shape/finiteness, temporal
identity, unchanged readout JSON and missing-input abstention. Full local suite:
**2,605 passed, 11 skipped**; Ruff passed.
