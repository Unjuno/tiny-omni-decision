# Path A four-modality dispatcher measurement — 2026-10-11

## Scope

This is a runtime-wiring and replay-parity check on already observed development
validation examples. It is not a new quality evaluation, checkpoint selection,
blind audit, or deployment qualification. No model training ran and the sealed
audit was not loaded.

One persistent dispatcher routes Text, Audio, Image, and Video requests to two
resident profiles:

- Torch/CUDA: Audio, Text, and Video candidate scoring. The matching MiniLM
encoder is consolidated to one resident instance after runtime construction.
- JAX/TensorFlow CPU: VideoPrism Image encoding and raw Video encoding.

Video uses the same pinned VideoPrism checkpoint already included for Image.
The JAX worker keeps a bounded, 64-entry content-hash LRU for immutable pooled
Image and Video features. A repeated observation can be scored for another
taxonomy or candidate without rerunning VideoPrism. The cache retains feature
vectors and hashes, not the source media. The JAX Image path encodes one image
at batch size one; its probability output was compared with the saved fixed
batch-of-eight validation runtime.

## Inputs and implementation

- Source branch and commit: `docs/pretrained-reuse-roadmap-20261010` /
  `24118c48ebde29732d4d2d85cac3fe9621bae217`.
- VideoPrism source commit: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Assembly: `path-a-modular-runtime-assembly-v4`; package manifest SHA-256
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Four task samples were read from the saved synthetic Text, Speech Commands,
  CLEVR-4, and CLEVRER development-validation records. Each warm request reused
  the same observation; Image asked another taxonomy, Video scored another
  candidate on the same scene, Text asked another question about the same
  normalized state, and Audio repeated its four-query request.
- Every media file was SHA-256 checked. The text examples were reconstructed
  with the repository's normalized dataset loader and validation manifest,
  preserving the saved option order.
- The original assembly was not changed. Its tree contained two unmanifested
  Python `__pycache__` files created by earlier runtime imports, so strict
  package verification rejected that tree. The runner made a clean view from
  the 42 files explicitly listed in the unchanged manifest using same-volume
  hard links (zero file bytes copied), verified every file/hash, and ran workers
  with bytecode writing disabled. The original manifest hash stayed unchanged.
- The script is `scripts/benchmark_path_a_dispatcher.py`; output is
  `C:/CodexArtifacts/pretrained-reuse-r2/path-a-dispatcher-v0-attempt-03/`.
  Its machine-readable report SHA-256 is
  `29c7fe4b98cf8ba084e32adb3572cb24f2b332f18279585ead72044ee135823e`.
- Implementation SHA-256 values are recorded in that machine-readable report
  for the dispatcher, benchmark runner, and both workers. The shared bounded
  LRU helper SHA-256 is
  `9759265a3cbd8a5b30ef940ead9da9be609f31051128a70e2e221ac1e1c3503e`.

Reproduction command from the repository root:

```powershell
python scripts/benchmark_path_a_dispatcher.py `
  --assembly 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4' `
  --videoprism-python 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  --videoprism-source 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
  --artifact-root 'C:\CodexArtifacts\pretrained-reuse-r2' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-dispatcher-v0-attempt-03' `
  --device cuda
```

The runner refuses to overwrite an output directory. The output includes the
exact validation IDs and input-file hashes, per-request predictions/deltas,
worker logs and profile metadata.

## Results

All first and warm requests matched the saved task predictions. Each probability
vector stayed within `1e-5` absolute error and every predicted option index was
identical.

| Task/request | First request | Warm request | Max probability delta | Warm feature-cache hit |
| --- | ---: | ---: | ---: | --- |
| Video; same CLEVRER scene, next candidate | 16.502 s | 0.0100 s | `1.20e-7` / `2.98e-8` | Yes |
| Image; same CLEVR-4 image, next taxonomy | 1.219 s | 0.00489 s | `1.28e-8` / `2.39e-7` | Yes |
| Text; same state, next question | 0.00795 s | 0.00646 s | `4.47e-7` / `4.47e-7` | Not applicable |
| Audio; four queries on one Speech Commands clip | 0.1793 s | 0.01165 s | `8.39e-9` for all 4 | Not applicable; encoder-side feature caching is not implemented |

The initial Video request included both worker starts: Torch profile startup
`6.384 s`, VideoPrism profile startup `3.789 s`, video decode `0.198 s`,
preprocessing `0.033 s`, VideoPrism forward `5.857 s`, and CUDA scorer forward
`0.209 s`. The Image call used batch size one and matched the saved prediction;
its warm taxonomy request reused the identical 768-value VideoPrism feature.
The Video warm request used a different candidate statement but the same
video hash and feature hash.

Runtime accounting:

- Host: Windows 10, Python 3.11.9 Torch worker and Python 3.11.16 VideoPrism
  worker.
- Torch: PyTorch `2.6.0+cu124`, Transformers `5.6.2`, RTX 3080 Laptop GPU.
- VideoPrism: JAX `0.6.2`, TensorFlow `2.18.1`, NumPy `1.26.4`, CPU device
  `TFRT_CPU_0`.
- Unique package parameters: `170,499,716`; package content bytes excluding
  root manifest: `687,254,772` (about 687.3 MB). The provisional 100–200M
  parameter target is met; the provisional 200 MB package target is not.
- Torch resident unique parameters after MiniLM consolidation:
  `55,986,819`. VideoPrism encoder: `114,365,184`.
- Combined worker RSS after load: `1,509,298,176` bytes. Torch CUDA allocation
  after load: `225,261,056` bytes; startup peak allocated before MiniLM
  consolidation: `364,396,032` bytes. VideoPrism peak working set after the
  video forward: `4,160,888,832` bytes.
- Total sequential benchmark wall time: `20.884 s`; no cloud compute or model
  training was used.

## Failed attempts retained

- `attempt-01` stopped at strict verification because the original assembly
  contained the two unmanifested generated bytecode files. No model worker was
  started. The fix uses a manifest-only hard-link view instead of weakening
  package verification or deleting files from the original assembly.
- `attempt-02` ran the models and matched Audio, Image, and Video, but Text did
  not match because the harness initially bypassed the normalized-example
  loader. The harness now uses the same manifest-backed normalization as the
  Text probe. The failed report and logs are retained.

## Interpretation and remaining gates

This confirms the dispatcher can load and route all four narrow task
components, reuse a VideoPrism feature across a second Video question or Image
taxonomy, and reproduce existing task-specific decisions across the two runtime environments. Four
examples do not establish modality quality, latency distributions, concurrent
service behavior, joint audiovisual reasoning, general speech, natural-image
understanding, or release readiness. The package is still over the provisional
size target. The existing candidate tasks and validation sets are observed
development data; no blind or product-quality claim follows from this replay.

R2 remains open. In-domain task coverage and task-matched quality gates need to
be defined before any new training or product selection. No shared Lens,
cross-modal alignment, model merge, precision change, or package reduction was
started in this stage.
