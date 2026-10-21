# Path A modular package integrity recheck (2026-10-21)

## Scope and result

Read-only verification of the existing `path-a-modular-runtime-assembly-v4`.
No package file was modified by the audit, no model was loaded, and no
quantization was performed.

- Package manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- 42 manifest-listed files, total `687,254,772` bytes excluding the root
  manifest; declared unique runtime parameter count is `170,499,716`.
- The pinned size audit rechecked every manifest-listed path, byte count, and
  SHA-256 and completed successfully. Report:
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-assembly-audit-01\report.json`,
  SHA-256 `7074d97ca25dd4169979a5414dfb5040d849c80fba8cd1dced1752e44e7f3265`.
- Strict `verify_modular_bundle` rejected the package because the current
  directory contains two unlisted bytecode cache files under the image runtime:
  `__pycache__/__init__.cpython-311.pyc` (289 bytes) and
  `__pycache__/videoprism_runtime.cpython-311.pyc` (17,338 bytes). All listed
  files passed before the verifier reported the extra paths. Their origin is
  unknown. They were preserved; no cleanup was attempted.
- To isolate those post-assembly files without touching the source, a separate
  manifest-only copy was created at
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-integrity-copy-01`.
  Its `package-manifest.json` has the same SHA-256 as the source manifest,
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`, and the
  strict verifier passes. It contains exactly the 42 listed files, totals
  687,254,772 bytes, and declares 170,499,716 unique parameters. The original
  assembly and its two unlisted cache files remain unchanged.
- The manifest still reports `package_target_met: false`: current size exceeds
  the provisional 200,000,000-byte budget by 487,254,772 bytes.

## Storage estimate (not a quantized artifact)

The audit also reports a **storage-only hypothetical INT8 estimate** of
`182,018,983` bytes, assuming all FP32 tensors in the three encoder weight
files become one INT8 value per element, with 256-element groups, one FP32
scale and one INT8 zero point per group. It preserves all non-FP32 tensors,
existing encoder-file overhead, and the rest of the package. This calculation
does not exercise a quantizer or inference kernel, and it reports no quality,
latency, RAM, or runtime compatibility. It must not be treated as evidence that
the package target is met. Parameter count stays `170,499,716`; storage
compression does not reduce parameter count.

## Reproduction

```powershell
python scripts/audit_path_a_package_size.py `
  --assembly C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4 `
  --output-report C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-assembly-audit-01\report.json
```

The size auditor verifies manifest-listed hashes but intentionally does not
enforce exact directory contents. The original package directory remains
`INCOMPLETE` under strict verification because of its two extra bytecode files;
the separate manifest-only copy is strict-verifier clean. Both preserve the
same model files and manifest. No claim is made about the unknown origin or
runtime effect of the source directory's bytecode files.

The isolated copy can be reproduced with:

```powershell
python scripts/rebuild_manifest_listed_package.py `
  --source C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-integrity-copy-01
```

The utility refuses existing destinations and verifies every source entry
before copying. Integration tests confirm that an unlisted source cache is
omitted while the source stays unchanged, and that tampered listed files are
rejected.
