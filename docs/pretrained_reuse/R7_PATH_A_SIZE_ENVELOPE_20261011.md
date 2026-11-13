# Path A Package Size Envelope Audit — 2026-10-11

## Scope

This is a read-only storage estimate for the existing local v4 Path A package. It does not quantize weights, run inference, measure quality, or test a device kernel. It is not an R7 quality or runtime pass.

The audit read the package manifest and tensor headers, verified every listed file's byte size and SHA-256, and estimated a hypothetical weight-only INT8 representation for F32 encoder tensors. For each 256-element group, the estimate adds one FP32 scale and one INT8 zero point. It retains non-F32 tensor payloads, file/container overhead, and all package content outside the three encoder weights unchanged.

## Reproducibility

- Source branch: `docs/pretrained-reuse-roadmap-20261010`
- Source commit: `24120be0f76e139688b5a98375f42a84b5930977`
- Runner: `scripts/audit_path_a_package_size.py`
- Runner SHA-256: `e00c8ad81a24c51fd17a07e54e818473e14d675c4a6a9905588401c6c50bac8a`
- Manifest SHA-256: `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`
- Result: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-size-envelope-v0-attempt-03\size-audit.json`
- Result SHA-256: `7b9d4dcde1531a1c7917ac26d7d645847ca7ad58c04f85daadbcd6fa49408600`
- Manifest-listed files checked: 42
- Training run: none; sealed audit: not accessed

Attempt 01 exposed a dtype-normalization bug: NumPy's `float32` spelling was not treated as equivalent to the safetensors `F32` spelling. Its estimate is invalid and was not used. The audit was corrected, tests were added, and attempt 03 records the corrected runner and source provenance.

## Measured package and estimate

| Quantity | Result |
|---|---:|
| Current package files, excluding root manifest | 687,254,772 B |
| Total parameters in manifest | 170,499,716 |
| Three encoder weight files | 678,106,456 B |
| Non-encoder package content retained | 9,148,316 B |
| F32 encoder values | 169,515,648 |
| Non-F32 encoder values | 512 I64 values (4,096 B) |
| Estimated INT8 payload | 169,515,648 B |
| FP32 group scales, group size 256 | 2,648,924 B |
| INT8 group zero points | 662,231 B |
| Retained encoder/container overhead | 39,768 B |
| Estimated package | 182,018,983 B (~182.0 MB decimal) |
| Margin to provisional 200,000,000 B target | ~17,981,017 B (~18.0 MB) |

Encoder weight files:

| Component | File bytes | F32 parameters |
|---|---:|---:|
| VideoPrism | 457,475,296 | 114,365,184 |
| Whisper tiny | 151,061,672 | 37,760,640 |
| MiniLM-L3 | 69,569,488 | 17,389,824 |

The estimate assumes scales and zero points are stored as described and does not include any new index, alignment, runtime, or kernel metadata beyond the retained current file/container overhead. The roughly 18 MB margin is therefore an envelope estimate, not a packaging guarantee.

## Interpretation and next gate

The audit identifies encoder precision as the dominant package-size opportunity. It does not establish that INT8 preserves any task's accuracy or calibration, can be loaded by a compatible backend, reduces peak memory, or improves latency on a target device. The provisional 100–200M parameter and <=200 MB package goals remain engineering targets; task-specific quality floors and target-device requirements have not been supplied.

R7 remains open. Before producing an artifact, define the deployment device and task-matched quality gates, then run an isolated INT8 candidate against the same approved validation tasks and the current Path A baseline. Measure package bytes, output quality/calibration, load behavior, peak memory, and latency on that device. Do not treat this theoretical estimate as a quantized package or as product readiness.
