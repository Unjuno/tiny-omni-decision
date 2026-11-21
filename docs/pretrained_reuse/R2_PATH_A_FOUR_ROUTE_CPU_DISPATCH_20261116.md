# Path A four-route CPU dispatch replay — 2026-11-16

## Result

The v4 modular package started all four typed routes on CPU. Video, Image, and
Text matched their saved development-runtime probabilities within the existing
`1e-5` tolerance; Audio returned valid typed JSON and the same argmaxes, but its
maximum probability delta was `6.1774e-4`, so the aggregate runtime parity gate
failed. The existing strict tolerance is unchanged. This replay is runtime and
serialization evidence, **not a task-quality evaluation**.

| Route | First request (s) | Warm request (s) | Max probability delta, first/warm | Argmax mismatches | Typed JSON |
|---|---:|---:|---:|---:|---|
| Video | 29.3266 | 0.0204 | 0 / 1.49e-7 | 0 / 0 | pass |
| Image | 1.5942 | 0.0049 | 1.28e-8 / 1.19e-7 | 0 / 0 | pass |
| Text | 0.0751 | 0.0457 | 5.66e-7 / 5.66e-7 | 0 / 0 | pass |
| Audio, four queries | 0.2461 | 0.0028 | 6.18e-4 / 6.18e-4 | 0 / 0 | pass |

Dispatch timing is for these fixed, previously observed examples only. Video's
first request includes cold worker/model startup; warm timing reuses the same
scene cache. These values are not latency guarantees for other inputs or
devices.

## Reproduction and integrity

- Source: `codex/pretrained-reuse-next-20261115` at
  `bc137e7a592ad8bbdab1a29104caf71c5b9ca42e`; source worktree was clean.
- The PowerShell process set `PYTHONPATH=<repo>\src`, `CUDA_VISIBLE_DEVICES=""`,
  `JAX_PLATFORMS=cpu`, and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, and
  `TF_NUM_INTEROP_THREADS` to `2`. Runtime `--device cpu` was also explicit.
  The full PowerShell setup and benchmark command were:

  ```powershell
  $env:PYTHONPATH = (Join-Path (Get-Location) 'src')
  $env:CUDA_VISIBLE_DEVICES = ''
  $env:JAX_PLATFORMS = 'cpu'
  $env:OMP_NUM_THREADS = '2'
  $env:MKL_NUM_THREADS = '2'
  $env:OPENBLAS_NUM_THREADS = '2'
  $env:TF_NUM_INTRAOP_THREADS = '2'
  $env:TF_NUM_INTEROP_THREADS = '2'
  & 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe' `
    scripts/benchmark_path_a_dispatcher.py `
    --assembly 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4' `
    --videoprism-python 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
    --videoprism-source 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
    --artifact-root 'C:\CodexArtifacts\pretrained-reuse-r2' `
    --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-cpu-four-route-runtime-20261116-attempt-02' `
    --device cpu
  ```
- The main process used Python 3.11.9, PyTorch 2.6.0+cu124, and Transformers
  5.6.2. The VideoPrism worker used the existing pinned venv with Python 3.11.16,
  JAX 0.6.2, TensorFlow 2.18.1, and NumPy 1.26.4; it selected `TFRT_CPU_0`.
  No CUDA allocations were reported. No package was installed or upgraded.
- Total sequential benchmark time, including cold worker startup, was 35.0698
  seconds. Combined resident-set size after model load was about 1.37 GB; no
  cloud or paid compute was used.
- Audio encoder revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; text
  encoder revision `4ca70771034acceecb2e72475f72050fcdde4ddc`; VideoPrism
  revision `be719a406d563b66f0ac969e7c94bab8e997c81a`. The runtime has
  170,499,716 unique parameters and the package manifest reports 687,254,772
  component bytes, still above the provisional 200 MB package target.
- Package manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Attempt 01 is retained as a failed environment setup: the system Python was
  mistakenly selected for the VideoPrism worker and did not have JAX. Attempt
  02 used the already provisioned VideoPrism venv and completed all routes.
- The requested `C:\CodexArtifacts\...` artifact path resolved to
  `D:\CodexArchiveRelocated\CodexArtifacts\...` on this machine. The canonical
  attempt-02 output is
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-cpu-four-route-runtime-20261116-attempt-02\dispatcher-benchmark.json`;
  its SHA-256 is
  `fb8ab5530569e057a144b9a591779b557d11c2be97fef62e664820f7dd296b59`.

The Audio difference is consistent with the previously recorded CPU FP32 vs
saved CUDA BF16 discrepancy in
[the 2026-11-15 Audio parity decomposition](R5_PATH_A_AUDIO_CPU_PARITY_DECOMPOSITION_20261115.md).
All four first and warm predictions retain their saved argmax, but that does
not resolve strict parity or establish accuracy. No tolerance, runtime package,
checkpoint, corpus, or sealed-audit data was changed or used. Training was not
performed.

## Next gate

Keep the four individual routes as a dispatch baseline. R2 still lacks a
rights-cleared, independently grouped natural AV task whose labels require both
inputs. Do not train or claim joint AV quality until that admission gate and
the matched unimodal / modality-perturbation controls are fixed.
