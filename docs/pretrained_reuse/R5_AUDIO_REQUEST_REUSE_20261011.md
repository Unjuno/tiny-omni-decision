# Audio feature reuse across requests

Date: 2026-10-11. Development runtime change; no training or quality selection.

## Change

Repeated `decide_many` calls previously re-encoded the same WAV for every
request. Each frozen runtime instance now keeps up to 64 single-event CPU
features in a content-addressed LRU. Different questions use the same feature;
different file bytes produce a new encoding. The optional expected media hash
is checked before cache lookup. A file that changes during encoding is rejected
before insertion. Capacity zero disables caching for matched comparisons.

Batch inference bypasses this cache: BF16 outputs depend on encoder batch shape,
so single-event features cannot substitute for batch-eight reference features.
The cache is scoped to a fixed model/processor/backend instance. It is not a
persistent cache, concurrent streaming API, or learned temporal state.

At 384 FP32 elements per feature, 64 entries store 98,304 tensor bytes on CPU,
plus Python objects/keys. Weights, feature precision and Decision readout are
unchanged. The worker reports cache hits, content/feature hashes, entry count,
and cumulative single-event encoder calls. The bundle builder includes the
cache dependency so standalone imports remain functional.

## Local measured comparison

On the local idle RTX 3080 Laptop, two sequential worker processes used identical
frozen weights and single-event preprocessing, first with capacity 0 and then
64. Eight existing validation audio assets each received two four-query
requests. Two further assets were reserved for the observation-update protocol.
This fixed order and small sample establish correctness and indicative latency,
not service-level speed guarantees or a new accuracy evaluation.

| Measure | Cache disabled | Cache 64 |
| --- | ---: | ---: |
| Encoder calls for eight first + eight repeated requests | 16 | 8 |
| Repeated requests, median end-to-end worker roundtrip | 13.191 ms | 2.960 ms |
| Repeated requests, observed range | 12.327–14.305 ms | 2.836–3.180 ms |
| Maximum probability delta against uncached single-event output | 0 | 0 |

All 32 repeated decisions (8 clips × 4 questions) had probability vectors
identical to the corresponding uncached output. First-call initialization was about 0.39 s
in each process and is excluded from repeated-request timings. The full two-worker
measurement took 14.54 s. On two additional assets all seven update checks passed:
new query reuse, content alias reuse, stale hash rejection, same-path/same-mtime
replacement, repeat replacement, and original-content restoration. Their six
successful predictions also matched uncached outputs exactly. Original media,
source prediction/config files, and weights were preserved; before/after hashes
are recorded. No sealed audit was read. Cloud cost: $0.

## Artifacts and reproduction

- Base code commit: `808a70efcef7120a0061802c19a9db2fa8d94e2b` plus the working
  patch. Exact modified runtime/worker/runner hashes are in `result.json`.
- Successful output: `C:/CodexArtifacts/pretrained-reuse-r2/audio-request-cache-attempt-02/`.
- `result.json` SHA-256: `ed1181956f93eeca61554ce01d774b8a77014015011a494d8b020659e3914035`.
- Attempt 01 is retained as a failed measurement: the comparison record omitted
  `question_type`, causing `KeyError` before any comparison result. The runner
  now retains that field from the existing task metadata.
- Standalone bundle: `C:/CodexArtifacts/pretrained-reuse-r2/path-a-audio-request-cache-bundle-v1/`.
  It contains 26 payload files, 227,103,552 payload bytes, or 227,110,420 bytes
  including its manifest. It retains the original task family bundle ID;
  this distinct path and manifest identify the new software snapshot.
- Bundle manifest SHA-256: `c7fb2d58df1be5cefdd9ad76feaaeb8e5590c94ea0bba8c6aedfe60c5fe79062`.
  All files passed hash/size/file-set validation after standalone execution.
- An external script loaded the new bundle using `python -I -B`, outside the
  repository, and confirmed the imported runtime path was inside the bundle.
  Two requests used one encoding and reproduced baseline probabilities exactly.
  `standalone-result.json` SHA-256: `a22c8aa953310d2244581fa420ec69e8c951cd3f50bb24b4ff15d3576137a3f9`.
  External `verify_standalone.py` SHA-256: `4c5cea5d973e720e6524136569a0db06956d3860bd82ac601018b669360f14cf`.
- Environment: Python 3.11.9, Torch 2.6.0+cu124, Transformers 5.6.2; model hashes
  and backend metadata are captured per worker in the result.

```powershell
python scripts/benchmark_path_a_audio_cache.py `
  --assembly C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4 `
  --artifact-root C:\CodexArtifacts\pretrained-reuse-r2 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\audio-request-cache-attempt-02
```

Use a new output directory on repetition. CPU regression tests cover content
reuse, stale hash rejection, capacity eviction, mutation during encoding, and
batch-cache isolation. This completes the missing sequential Audio reuse path;
task quality, joint AV contribution, deployment size and release gates remain open.
