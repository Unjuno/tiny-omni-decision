# Path A audio precision parity diagnostic (2026-11-14)

## Question and scope

The fixed CPU four-route qualification had one audio request exceed its saved
CUDA-reference probability tolerance of `1e-5` (maximum absolute delta
`0.000617743`). This one-input diagnostic tested whether the discrepancy was
primarily the documented CUDA BF16 autocast versus CPU FP32 precision path.
It used no labels, quality metrics, training, tuning, checkpoint selection, or
sealed-audit data. The input, model weights, and readout were read-only.

## Pinned inputs and execution

- Runtime source: `codex/path-a-canonical-json-runtime-20261114`, commit
  `6d722c258f188a0a09c624bf7b39e3f352d55319`.
- Device: NVIDIA GeForce RTX 3080 Laptop GPU.
- Python `3.11.9`, PyTorch `2.6.0+cu124`, Transformers `5.6.2`.
- Audio weights SHA-256:
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`;
  readout SHA-256:
  `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.
- Input media SHA-256:
  `77ad54371520bd5381b7758694bc6d98193e8f8864750ea56ea7f8085f2daaa1`.
- Preserved output:
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-cuda-fp32-parity-20261114-attempt-12\result.json`
  (SHA-256 `D32F02A5AEA997F9B48A397E8276C40360DB0462AA8CDEA25D95D8872E96DA4D`).
- Three preceding harness starts failed before model inference and remain in
  the attempt directory; they are not runtime results.

The same four audio decisions were run through normal CUDA BF16, CUDA FP32
(autocast disabled only within the diagnostic process), and CPU FP32. Normal
CUDA BF16 reproduced the saved CUDA outputs exactly. All three paths selected
the same argmax for all four decisions.

| Comparison | Maximum absolute probability delta |
|---|---:|
| CUDA FP32 vs saved CUDA BF16 | `0.000596762` |
| CPU FP32 vs saved CUDA BF16 | `0.000617862` |
| CUDA FP32 vs CPU FP32 | `0.000021100` |

The maximum is dominated by the 10-option query; the other three binary
queries had CUDA-FP32-vs-CPU-FP32 maximum deltas at or below `5.13e-6`.
CUDA FP32 took `0.628 s`, CPU FP32 `0.473 s`, and normal CUDA BF16 `3.165 s`
for this small process invocation; these are not stable throughput benchmarks.
Peak CUDA allocated memory was `484,143,616` bytes.

## Interpretation and decision

For this fixed input, CPU FP32 is numerically much closer to CUDA FP32 than to
the saved CUDA BF16 output. This supports precision/autocast mismatch as the
main explanation for the earlier audio parity failure, rather than a broad CPU
backend defect. It does **not** prove corpus-wide parity, change the existing
`1e-5` reference gate, or establish any task-quality result. The four argmaxes
agree, but probability parity still fails the strict saved-BF16 threshold.

No runtime or model change is justified from one audio item. Preserve the
existing BF16 CUDA product path and CPU FP32 path. If CPU probability parity is
required, first define a precision-matched reference contract and repeat a
bounded fixed-input parity check across representative audio shapes before
changing runtime code. No GPU training or live-process intervention occurred.
