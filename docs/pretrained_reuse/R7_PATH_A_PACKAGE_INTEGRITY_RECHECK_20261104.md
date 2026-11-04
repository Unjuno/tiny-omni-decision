# Path A v4 package integrity recheck — 2026-11-04

## Result

Read-only audit of the existing Path A v4 modular assembly and its previously
created manifest-only integrity copy. No source package file was edited, no
model weights were loaded, no inference or quantization was run, and the sealed
audit was not accessed.

- Source commit: `a93b9faae21fcc4c99f2766cb72fba5ea43a8761`.
- Package manifest SHA-256 in both directories:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Every one of the 42 manifest-listed payload files passed byte-size and
  SHA-256 validation in the size audit.
- Strict `verify_modular_bundle` passes on the preserved integrity copy:
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-integrity-copy-01`.
- Strict verification still fails on the original assembly because it has two
  unlisted bytecode files, unchanged since 2026-10-11:
  `components/image/runtime/tiny_omni_decision/__pycache__/__init__.cpython-311.pyc`
  (289 bytes) and `videoprism_runtime.cpython-311.pyc` (17,338 bytes). Their
  provenance remains unknown; both are preserved.
- Manifest accounting: 170,499,716 unique runtime parameters and
  687,254,772 payload bytes (root manifest excluded). The provisional 200 MB
  package target is still not met by the actual package.
- The audit's storage-only INT8 estimate is 182,018,983 bytes under its stated
  assumptions (group size 256, FP32 scales, INT8 zero points; all other payload
  and file overhead retained). This is not a quantized artifact or a runtime,
  quality, memory, or latency measurement.
- External audit report SHA-256:
  `5c36ccf1c71bec7151a93d46cfb6c122d8d315ecd719b2689d7de32716d1d531` at
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-assembly-audit-20261104-01\report.json`.
- C: had approximately 5.16 GiB free after the audit.

## Gate status

This confirms current payload integrity and exact manifest accounting only. It
does not make Path A a release candidate: the established task-quality gate is
not met, the true package remains above 200 MB, quantized save/reload parity is
unmeasured, and the user device/latency-energy/task battery remains open. R7
quantization remains deferred until the quality gate is satisfied.
