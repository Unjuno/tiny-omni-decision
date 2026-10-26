# Path A package integrity recheck — 2026-10-26

## Result

The existing v4 Path A assembly failed the strict package verifier because its
directory had two extra Python bytecode files that were not in the immutable
package manifest. All 42 manifest-listed files were present with the recorded
sizes and SHA-256 hashes; no listed file was missing. The extra files were
`components/image/runtime/tiny_omni_decision/__pycache__/__init__.cpython-311.pyc`
(289 bytes) and
`components/image/runtime/tiny_omni_decision/__pycache__/videoprism_runtime.cpython-311.pyc`
(17,338 bytes).

The original assembly was left unchanged. The existing read-only
`rebuild_manifest_listed_package.py` helper copied only the 42 manifest-listed
files plus the root manifest into a new directory, then reran strict bundle
verification successfully:

`C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-clean-copy-20261026-01\`

The source and copy both retain root-manifest SHA-256
`79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
The clean copy contains exactly 42 manifest-listed content files and no
unlisted files. This is a byte-for-byte package-integrity repair in a new
directory; no model was loaded, no inference/training was run, and no
checkpoint, corpus, validation output, sealed audit, or original package was
changed.

## Reproduction and gate status

From the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
python scripts/rebuild_manifest_listed_package.py `
  --source 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-clean-copy-20261026-01'
```

The strict verifier returned `PASS_MANIFEST_LISTED_PACKAGE_COPY` with
`strict_verifier_passed=true`, `unique_runtime_parameters=170,499,716`, and
`package_file_bytes_excluding_root_manifest=687,254,772`. The provisional
200 MB package target remains unmet. This closes only strict package-file-set
integrity for the clean copy; it does not add runtime environments, reduce
package size, or qualify model quality or end-user deployment.

The preserved original remains at
`C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\`
with its original manifest hash. Its two unmanifested bytecode files are still
there and were not deleted or modified.
