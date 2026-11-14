# Path A CPU fallback runtime recheck — 2026-11-14

## Scope and disposition

This is a small CPU-only runtime feasibility and saved-output parity diagnostic
for the existing four-route Path A JSONL runtime. It is not a quality
re-evaluation, a latency service-level benchmark, a training run, or release
qualification. It used two distinct already-observed development-validation
assets per route (eight unique assets total), plus same-asset warm requests.
No sealed audit or new labels were loaded. The prior checkpoint assembly and
validation inputs were read-only.

Overall status is **`failed_parity`** because the Audio route exceeded the
existing `1e-5` probability parity threshold on both samples. Text, Image, and
Video each passed for both samples. All eight typed JSON outputs were internally
consistent with the runtime option order and predictions. Audio's argmax did
not change, but probability-vector parity did; this is a numerical runtime
mismatch, not evidence of task-quality change.

## Reproduction

- Source branch/commit: `codex/path-a-canonical-json-runtime-20261114` /
  `6d722c258f188a0a09c624bf7b39e3f352d55319`.
- Runner: `scripts/benchmark_path_a_latency_distribution.py`.
- Command:

  ```powershell
  python scripts/benchmark_path_a_latency_distribution.py `
    --assembly 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4' `
    --videoprism-python 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
    --videoprism-source 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
    --artifact-root 'C:\CodexArtifacts\pretrained-reuse-r2' `
    --output-dir 'D:\CodexArtifacts\pretrained-reuse-r2\path-a-canonical-cpu-runtime-v0-attempt-01' `
    --samples-per-modality 2 --device cpu
  ```

- Result JSON: `D:\CodexArtifacts\pretrained-reuse-r2\path-a-canonical-cpu-runtime-v0-attempt-01\latency-distribution.json`.
- Result SHA-256: `a6c84fe72c11f411fafebf31b067541fc944786ac0c788dcce7e8b8aabde99b2`.
- Assembly manifest SHA-256: `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- VideoPrism source revision: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- VideoPrism model revision: `be719a406d563b66f0ac969e7c94bab8e997c81a`; checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Torch Audio encoder revision: `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; weights SHA-256:
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Host/runtime: Windows 10 build `10.0.26300`; Intel Core i7-12700H (14 cores / 20 logical processors); Python 3.11.9; PyTorch `2.6.0+cu124` with requested device `cpu`; Transformers `5.6.2`; VideoPrism Python 3.11.16, JAX `0.6.2`, TensorFlow `2.18.1`, NumPy `1.26.4`, device `TFRT_CPU_0`.
- Workers reported 55,986,819 unique Torch-resident parameters and 114,365,184 VideoPrism parameters. Combined worker RSS after load was 1,392,500,736 bytes; VideoPrism peak working set was 4,153,421,824 bytes.
- Total sequential time: 49.021 seconds. No GPU inference, training, cloud compute, or package upgrades were used. The GPU remained occupied by the unrelated Orbitune process and was not touched.

## Measurements

| Route | Assets | First request median (s) | Same-asset warm median (s) | Saved output parity | Max probability delta |
|---|---:|---:|---:|---:|---:|
| Text | 2 states | 0.102 | 0.097 | 2/2 pass | `5.66e-7` |
| Image | 2 images | 1.575 | 0.0071 | 2/2 pass | `1.28e-8` |
| Audio | 2 clips / 8 queries | 0.321 | 0.0035 | 0/2 pass | `6.18e-4` |
| Video | 2 scenes | 18.272* | 0.029 | 2/2 pass | `8.94e-8` |

`*` Video first-request median includes worker cold start and XLA initialization.
The first video call took 34.857 s (Torch worker start 10.287 s, VideoPrism
worker start 15.535 s, VideoPrism forward 8.486 s); the next distinct video
request took 1.687 s. With only two assets, p90 is the observed maximum and is
not a reliable service-tail estimate. The runner used cached VideoPrism
features for the warm same-video candidate requests and image taxonomy
requests. Text and Audio have no equivalent encoder-feature reuse in this
runtime.

Audio deltas by sample were `0.0006177425` and `0.0000744720`; both exceed
`1e-5`. The maximum remains below the previous CPU typed-JSON replay's reported
`0.000617743`, but the current run independently confirms the gate is still
failing. The source/reference cause has not been isolated, and no tolerance was
changed. Keep CPU Audio probability parity **open** until the CPU and reference
numerical path is reconciled.

## Interpretation and next gate

This diagnostic establishes that the existing runtime can execute all four
routes on CPU with low resident memory after load and that three routes match
the saved reference on these two examples each. It does **not** establish
broad CPU latency, target-device suitability, task quality, or a release-ready
CPU fallback. Video's cold initialization is expensive; Audio's saved-output
probability parity is unresolved. The current 170,499,716-parameter,
687,254,772-byte full-precision assembly accounting remains unchanged; this
measurement does not pass the provisional 200 MB package objective.

The next bounded action is a CPU-versus-reference numerical-path investigation
for Audio using the same already-saved predictions, without fitting, changing
the checkpoint, or touching validation labels. If parity is fixed, repeat a
larger latency sample only when a concrete deployment requirement needs it.
No shared Lens, model merge, quantization, or training is justified by this
runtime-only result.
