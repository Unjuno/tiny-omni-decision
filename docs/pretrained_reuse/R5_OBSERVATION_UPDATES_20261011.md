# Path A observation replacement and reuse

Date: 2026-10-11. This is a runtime correctness experiment on existing
development examples, not a quality evaluation. Frozen weights and original
media were unchanged. No training or sealed audit access occurred.

## Protocol and outcome

The runner selected the first two distinct Image assets and Video scenes from
the same saved development predictions used by the dispatcher benchmark. Each
task used a separate writable media copy. The predeclared gate was matching
prediction indices, maximum absolute probability delta <=1e-5, and the expected
encoder-cache hit/miss for each operation. A stale expected content hash had to
raise the specific frozen-identity error. Unexpected errors fail the probe.

| Operation, for each modality | Expected | Observed |
| --- | --- | --- |
| First observation A | Encode | PASS |
| Different saved question on A | Reuse | PASS |
| A copied to a different filename | Reuse | PASS |
| Replace A with B at the same path, keep A's timestamp, supply A's hash | Reject | PASS |
| Request B at the same path and timestamp with B's hash | Encode | PASS |
| Request B again | Reuse | PASS |
| Restore A's bytes at the original path | Reuse | PASS |

All 14 checks passed. Twelve successful decisions used four encoder executions
and eight cache hits. Both stale-hash requests were rejected before inference.
Every successful decision matched its saved prediction index; the largest
probability delta was 2.3841858e-7 for Image and 1.1920929e-7 for Video. Hashes
of all source media and reference files matched before and after the run.

The complete run took 22.89 seconds. Encoding replacement B took 1.016 s for
Image and 1.168 s for Video; repeat B requests took 6.09 ms and 10.23 ms,
respectively. These individual observations include dispatcher overhead and
are not latency-distribution estimates. The first Image/Video requests took
9.979/7.804 s including their relevant worker startup/initialization.

## Reproduction

- Source commit: `40c5b3965b7db897f59bcd74e408278b1b694b42`, plus the new runner
  SHA-256 `0648b6847c76c1f62d91c5878828a7698fa55474f66f1d6069792690e2e6c0c9`.
- Run: `C:/CodexArtifacts/pretrained-reuse-r2/path-a-observation-updates-attempt-01/`.
- `result.json` SHA-256: `04504941bbe75b0169d7ddf93cdd2b19cfe62c0a46c6855dc34321a0fe2abe1e`.
- Full decisions, cache metadata, per-request resources and errors are saved
  incrementally in `events.jsonl`; selected source IDs/hashes, environment,
  model revisions and command are in `result.json`.
- The existing v4 assembly was accessed through a strictly verified manifest
  view; original package files were not edited. Torch scoring used the local
  RTX 3080 Laptop, Python 3.11.9, Torch 2.6.0+cu124, Transformers 5.6.2.
  VideoPrism encoding used CPU, Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1.
  GPU was idle before launch; only unrelated Django Python processes existed.
  Cloud cost: $0.

```powershell
python scripts/verify_path_a_observation_updates.py `
  --assembly C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4 `
  --artifact-root C:\CodexArtifacts\pretrained-reuse-r2 `
  --videoprism-python C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe `
  --videoprism-source C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\path-a-observation-updates-attempt-01 `
  --device cuda
```

Use a new output directory on repetition. The runner refuses to overwrite.
CPU regression tests confirm the probe rejects path-only cache identity and
acceptance of stale hashes, and preserves original fixture media.

## Limits and next implementation need

This verifies sequential file replacement, content identity and multi-query
reuse for two visual assets per task. It does not test concurrent file writes,
stream timestamps, causal state updates, long-term memory, or sensor fusion.
An old feature can remain resident under its old hash without being served for
new content; this is intentional reuse, not deletion of prior observations.
Image/Video use a fixed 64-entry in-process cache. At this measurement, Audio
reused one encoding within a query group, but recomputed it on a later request
for the same file. The subsequent cross-request Audio reuse implementation and
matched uncached comparison are recorded in `R5_AUDIO_REQUEST_REUSE_20261011.md`.
No R2 quality, R7 package, or R8 release gate is closed by these measurements.
