# Path A Audio CPU/CUDA parity diagnosis — 2026-11-14

## Scope

This was a read-only inference diagnostic for the existing fixed Path A Audio
request. It used the already pinned Whisper Tiny, MiniLM and Audio readout from
the manifest-only assembly view, and compared returned probabilities with the
saved canonical JSONL CUDA output. It did not inspect labels, calculate task
quality metrics, fit or select a model, alter the assembly, access the GPU, or
change the existing `1e-5` parity tolerance.

## Results

| CPU execution policy | Audio decisions | Maximum absolute probability delta vs saved CUDA output | `1e-5` parity |
|---|---:|---:|---|
| FP32, current production CPU path | 4 | `6.1774254e-4` | Fail |
| BF16 autocast, diagnostic only | 4 | `6.6041946e-4` | Fail |

The FP32 replay reproduces the prior CPU dispatcher finding (`6.1774254e-4`).
Applying CPU BF16 autocast as a diagnostic did not solve the mismatch and was
slightly farther from the saved CUDA probabilities. Therefore no precision
policy change is justified by this sample. The CPU FP32 path remains unchanged;
cross-device equality at `1e-5` remains unresolved. This evidence does not
establish a quality difference or a task failure.

## Reproducibility

- Source worktree: `codex/path-a-audio-cpu-parity-20261114`, based on
  `6d722c258f188a0a09c624bf7b39e3f352d55319`.
- Interpreter: Python `3.11.9`; PyTorch `2.6.0+cu124`; Transformers `5.6.2`.
- Device: CPU. `CUDA_VISIBLE_DEVICES` was empty; worker metadata reported no
  CUDA allocation. The active GPU process was not queried for control or
  interrupted.
- Assembly: existing manifest-only view from dispatcher attempt 01; the
  canonical attempt 07 request and output were read-only.
- Audio weight SHA-256:
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Audio readout SHA-256:
  `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.
- Saved reference output SHA-256:
  `0cd3437ecd32310a724f2d982c641c7497ff88bab3205c01f07c79e7aebf675a`.
- CPU FP32 result: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-cpu-parity-20261114-attempt-09\cpu-fp32-baseline.json`.
- CPU BF16 diagnostic: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-cpu-parity-20261114-attempt-11\cpu-bf16-diagnostic.json`.
- CPU FP32 result SHA-256: `963edd7634a94046d0bca1d95f25d73b014727057681351626bc1e58e4a0ecce`.
- CPU BF16 diagnostic SHA-256: `12010b90b560e6a6e206eba93bfe5b441c9bd4450ae0ebd13fe516182fd85598`.

The initial diagnostic harness attempts 08 and 10 stopped before scoring due to
harness setup errors (reference JSONL array shape; then an absent optional
media-hash field). They did not touch model state or the GPU; successful runs
09 and 11 used the corrected harness and preserved independent output paths.

## Interpretation and next action

The mismatch is reproducible, but the two tested CPU precision modes do not
make CPU outputs match the saved CUDA route at the existing threshold. The
available evidence does not isolate the difference to one kernel or prove
that CPU FP32 is incorrect. Keep exact weights and task-quality evaluation
separate from this device-portability issue. If strict cross-device parity is
required, first establish a CPU-specific deterministic reference and compare
within-device repeats; do not silently widen the CUDA parity threshold.

This diagnostic closes no R2 task-admission or multimodal-quality gate. The
next quality-bearing work remains selecting a rights-cleared, independently
grouped task for which both input modalities are necessary, then freezing its
protocol and controls before evaluation.
