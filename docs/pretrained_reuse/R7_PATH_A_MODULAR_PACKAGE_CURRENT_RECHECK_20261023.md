# Path A modular package current recheck — 2026-10-23

## Result

A fresh read-only audit checked all 42 manifest-listed files in the existing
`path-a-modular-runtime-assembly-v4`. Every listed file's recorded byte count
and SHA-256 matched. The current manifest hash is
`79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.

| Check | Result |
|---|---:|
| Manifest-listed files | 42 |
| Declared unique runtime parameters | 170,499,716 |
| Package bytes, excluding root manifest | 687,254,772 |
| Provisional 200,000,000-byte package target | **Not met** |
| Storage-only hypothetical groupwise INT8 estimate | 182,018,983 bytes |
| Quantized weights / inference kernel / quality evaluated | No / no / no |

The 182 MB figure remains only a storage estimate. It does not show that an
INT8 package exists, preserves any Decision metric, loads in the runtime, or
reduces latency or memory.

## Strict directory verification

`verify_modular_bundle` still rejects the original v4 directory because the
image runtime contains two files outside the manifest:

| Unlisted file | Bytes | SHA-256 |
|---|---:|---|
| `components/image/runtime/tiny_omni_decision/__pycache__/__init__.cpython-311.pyc` | 289 | `02dd90f87c7e557f38e649f32b46d13d5c8d12c5164e46df2d4751a65c3b948f` |
| `components/image/runtime/tiny_omni_decision/__pycache__/videoprism_runtime.cpython-311.pyc` | 17,338 | `70f4f9053df55bc3cb78340efa84df87ac3aba2fb74060e62e88551e5d0c65e1` |

The existing manifest-only integrity copy passes strict verification and
reports the same 170,499,716 parameters. The original package and the copy were
left unchanged. Existing image runtime parity evidence remains limited to
replaying saved development predictions; it is not a new quality result.

## Reproducibility and limits

- Repository commit: `fda4e7f3e7b191bf20496477e104df8fe118bd08`.
- Runner: `scripts/audit_path_a_package_size.py`.
- Runner SHA-256: `b0d736c3c46feb17a6c810cc1ead461780a0e72a192a8ef140aaf8a749f1976d`.
- Audit report: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-assembly-audit-02\report.json`.
- Report SHA-256: `562eccfb15f3ba3a4f41e1565a689d152ca18329021765018dabf95250738b5e`.
- The report is a read-only audit and storage estimate. No model was loaded,
  trained, quantized, or run; no sealed audit was accessed; no GPU was used.

R7 remains open. Before a quantized candidate can be considered, it needs an
actual separately stored representation, an inference path, and task-matched
quality/calibration, memory, latency and package-size measurements. Full
four-modality product readiness remains unproven.
