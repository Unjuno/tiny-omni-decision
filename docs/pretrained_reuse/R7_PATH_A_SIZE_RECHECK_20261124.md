# Path A package size recheck — 2026-11-24

## Result

Re-ran the repository's read-only package audit against the existing Path A v4 assembly. All 42 manifest-listed files matched their recorded byte counts and SHA-256 values. The assembly still contains **170,499,716 unique runtime parameters** and **687,254,772 bytes** excluding its root manifest. The package-size gate of 200,000,000 bytes still fails.

The storage-only groupwise INT8 estimate remains **182,018,983 bytes**, about 17.98 MB under the nominal limit. This remains an estimate: no quantized weights, inference kernel, quality comparison, load test, memory measurement, or latency measurement was produced. The package manifest explicitly says it does not support joint audio-video reasoning. It is not a qualified Omni product artifact.

The separate native common-fit accounting is **not this package**: it has a different model inventory and 196,254,874 total parameters. Do not combine those figures with the v4 package's parameter or byte totals.

## Reproduction and outputs

- Branch and source commit: `codex/r2-bert-text-source-screen-20261124`, `0815eea285314d4c234826836ed7284d494eebc3`.
- Runner: `scripts/audit_path_a_package_size.py`, SHA-256 `b0d736c3c46feb17a6c810cc1ead461780a0e72a192a8ef140aaf8a749f1976d`.
- Assembly manifest SHA-256: `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- C-drive report: `C:\CodexArtifacts\path-a-size-recheck-20261124-attempt-02\size-audit.json`, SHA-256 `fdfabc7709b7f2c54ccb834450b6d8c3f5f39b53457b9c4560d4223807ce9c65`.
- The report was written on the C: volume. The input path `C:\CodexArtifacts\pretrained-reuse-r2\...` is a junction to `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2`; the existing assembly was read in place and not moved or modified.
- A first audit invocation used a new output directory beneath that junction and therefore wrote a second small report to D:. It was left intact. The C-drive attempt above is the authoritative recheck artifact.
- The audit was read-only with respect to the package. It did not load a model, use the GPU, access sealed audit data, quantize weights, or alter existing run/checkpoint files.

## Interpretation

R7's size estimate is reproducible, but R7 is not passed. A real INT8 candidate still needs a new output directory on the C: volume, an approved validation comparison that preserves task-specific quality, backend load/parity checks, and measured target-device memory and latency. The current task-quality and joint-input gates remain open, so this audit does not justify product promotion or compression-driven quality claims.
