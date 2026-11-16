# Path A Audio intermediate precision diagnosis — 2026-11-17

## Result

The initial CPU/CUDA Audio probability mismatch was localized to numerical
differences in the Whisper encoder path. A fixed, already-observed development
request was run with identical processor inputs under CPU FP32, CUDA BF16, and
CUDA FP32. Input feature tensors, token IDs, and attention masks matched
exactly across devices. Text encoder differences were small. The initial CUDA
FP32 versus CPU FP32 probability difference was just above the existing strict
runtime gate. A follow-up then disabled both CUDA matmul and cuDNN TF32 only
around the Whisper forward; it brought the same fixed request within the gate,
including through the packaged dispatcher. This is a four-query runtime
diagnostic, not quality evidence or a broad cross-device parity claim.

| Compared stage vs CPU FP32 | CUDA BF16 max / mean absolute delta | CUDA FP32 max / mean absolute delta |
|---|---:|---:|
| Whisper input features | 0 / 0 | 0 / 0 |
| Whisper encoder hidden state | 10.2515 / 0.01230 | 0.1555 / 0.0000912 |
| Pooled Audio feature | 0.08696 / 0.006316 | 0.004124 / 0.0002239 |
| Text encoder hidden state | 1.907e-6 / 1.694e-7 | 1.907e-6 / 1.694e-7 |
| Pooled Text features | 5.066e-7 / 8.844e-8 | 5.066e-7 / 8.844e-8 |
| Pair feature matrix | 0.08696 / 0.001976 | 0.004125 / 0.00006984 |
| Option logits | 0.06669 / 0.03107 | 0.001569 / 0.0007204 |
| Probabilities | 6.177e-4 / 9.080e-5 | 2.098e-5 / 3.553e-6 |

The fixed request contains four queries (10, 2, 2, and 2 choices). The initial
CUDA FP32 maximum probability delta was `2.0981e-5`; the follow-up's direct
CUDA FP32-no-TF32 path reduced it to `1.1921e-7`. The actual dispatcher worker
then processed the same four queries with `fp32_no_tf32`, returned 16 option
probabilities and four typed decisions, and measured maximum probability delta
`1.1921e-7` against CPU FP32, below the unchanged `1e-5` threshold. This only
passes for this fixed request on the recorded software/GPU stack. It does not
establish general cross-device parity or Audio quality. No labels were opened
and no Accuracy, NLL, Brier, or ECE was computed.

| Follow-up dispatcher measurement | Result |
|---|---:|
| Worker inference (four queries) | 0.3619 s |
| End-to-end dispatch including startup | 6.4394 s |
| Worker startup | 6.0497 s |
| Peak CUDA allocated since consolidation | 263,429,120 bytes |
| RSS after request | 1,252,635,288 bytes |
| Max probability delta vs CPU FP32 | 1.1921e-7 |
| Probability gate / typed JSON | pass / pass |

## Implementation

`SpeechCommandsMultiQueryRuntime` accepts `audio_encoder_precision`: `auto`,
`bf16`, `fp32`, or `fp32_no_tf32`. The default `auto` preserves the existing
behavior (BF16 autocast on CUDA, FP32 on CPU). `fp32_no_tf32` saves and
temporarily disables CUDA matmul and cuDNN TF32 flags around only the Whisper
forward, then restores both flags in `finally`. The dispatcher and Torch worker
pass the explicit choice through, and worker metadata records the effective
precision. No model weights, processor, readout, default precision, selector,
or task data changed. This opt-in runtime control is not a product precision
decision.

## Reproduction

- Source branch: `codex/pretrained-reuse-next-20261115`.
- Diagnostic source commit: `640715d4f08a973b2329511721d8fc33b11449dc`.
- Diagnostic script SHA-256: `449a4a0e04d5b0d649b11820a961bb0122ff1ecb77357da3c54b033025275c07`.
- Fixed request: prior Path A dispatcher attempt 07; request JSONL SHA-256
  `ae200aa3c099a2d8ead6d1dd1890122cf8b093ac1989fe0876f69735b267123e`.
- Audio SHA-256:
  `77ad54371520bd5381b7758694bc6d98193e8f8864750ea56ea7f8085f2daaa1`.
- Assembly manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Whisper revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; MiniLM revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; the existing Audio readout
  SHA-256 is `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.
- Command:

  ```powershell
  $env:PYTHONPATH = (Join-Path (Get-Location) 'src')
  $env:OMP_NUM_THREADS = '4'
  $env:MKL_NUM_THREADS = '4'
  $env:CUDA_VISIBLE_DEVICES = '0'
  & 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe' `
    scripts/diagnose_path_a_audio_precision.py `
    --run-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-canonical-dispatcher-20261114-attempt-07' `
    --assembly 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4' `
    --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-stage-precision-20261116-attempt-03'
  ```

- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  NVIDIA RTX 3080 Laptop GPU. CUDA inference-only peaks were 251,823,104 bytes
  allocated / 283,115,520 reserved for BF16 and 261,176,320 / 295,698,432 for
  FP32. Inference after model load took 0.4925 s and 0.07085 s, respectively;
  this one-request timing is diagnostic and is not a latency comparison.
- Initial direct precision result JSON SHA-256:
  `ba74e29a0a363d95f609e8c636a3f55fd1ffcc62bba996d07eae4dbb77106dc0`.
- Saved intermediate archives (processor tensors, encoder outputs, pooled
  features, pair features, logits, and probabilities):
  `cpu_fp32-intermediates.npz` SHA-256
  `a156029cd817a3189eb7d146d63918d3894e2981218e641369432b06d4777483`;
  `cuda_bf16-intermediates.npz` SHA-256
  `0acb8ac38523740251164e74dda29dc94e75788e4cce1c32c2eb541f61f55999`;
  `cuda_fp32-intermediates.npz` SHA-256
  `322217e5d5be86bff390b7e418ab214aac9ce4a474024640308d27f11d194733`.
- Final artifacts are under
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-audio-stage-precision-20261117-attempt-06\`.
  Follow-up JSON SHA-256:
  `36686afe5bcfbede918426032483df444f2316a5145a41bb183e70cdb7b2516f`;
  `cuda_fp32_no_tf32-intermediates.npz` SHA-256
  `ca670141ec14135e399840dc543893a1c37b1eb41b53fc1f81220bafbf18c4db`.
- Follow-up source commit from run metadata: `e7ca13659cc2a190ce562d37f4bc4797c97e78f0`;
  diagnostic script SHA-256 `c9dae6deb950e08331eab273a1a1df94efb109c894982e1ad2c94a0380df6a88`.
  The command used the same fixed `--run-dir` and `--assembly` shown above, with
  `--output-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-audio-stage-precision-20261117-attempt-06'`.
- Follow-up runtime: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  RTX 3080 Laptop GPU. The dispatcher reported 55,986,819 resident unique
  parameters; Audio/Text weights and readout hashes are recorded in its
  metadata (Whisper `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`,
  MiniLM `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`,
  Audio readout `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`).
- Earlier attempts 01 and 02 and complete intermediate run 03 remain retained;
  attempt 06 is the complete package-dispatch follow-up. CPU tests after the
  implementation change: 24 passed across `test_audio_runtime.py` and
  `test_runtime_dispatch.py`; Ruff checks passed for changed Python files.
- The separate Orbitune PID 28376 was observed before and after this short
  inference. It remained running; it was not stopped or modified.

## Decision and next step

Keep `auto` as the default and keep the strict `1e-5` parity gate. For this
fixed request, the explicit `fp32_no_tf32` path passes the unchanged gate, so
there is no remaining mismatch to explain for these four queries. Do not
generalize that result to other recordings or claim general cross-device
parity. Before making this precision mode the supported cross-device contract,
compare saved per-example outputs on a predeclared, label-blind set of diverse
audio durations/inputs, verify backend flags are restored after normal and
exceptional forwards, and measure latency on the target device. Keep quality
and package-size gates separate; this runtime diagnostic establishes neither.
