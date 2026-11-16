# Path A Audio intermediate precision diagnosis — 2026-11-17

## Result

The existing CPU/CUDA Audio probability mismatch is introduced primarily by
the Whisper encoder's numeric path. A fixed, already-observed development
request was run with identical processor inputs under CPU FP32, CUDA BF16, and
CUDA FP32. Input feature tensors, token IDs, and attention masks matched
exactly across devices. Text encoder differences were small. Whisper hidden
states and pooled Audio features changed with device/precision, and the final
CUDA FP32 versus CPU FP32 probability difference remained just above the
existing strict runtime gate.

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

The fixed request contains four queries (10, 2, 2, and 2 choices). CUDA FP32
is substantially closer to CPU FP32 than CUDA BF16 for these queries, but the
maximum probability delta, `2.0981e-5`, still exceeds the unchanged `1e-5`
parity requirement. This does not establish cross-device parity generally.
No target labels were opened and no Accuracy, NLL, Brier, or ECE was computed.

## Implementation

`SpeechCommandsMultiQueryRuntime` now accepts an `audio_encoder_precision`
setting: `auto`, `bf16`, or `fp32`. The default `auto` preserves the existing
behavior (BF16 autocast on CUDA, FP32 on CPU). The dispatcher and Torch worker
pass the explicit choice through, and worker metadata records
the effective Audio precision. No model weights, processor, readout, default
precision, selector, or task data changed. This is a diagnostic/runtime
control, not evidence that FP32 is the best product setting.

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
- Final result JSON SHA-256:
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
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-audio-stage-precision-20261116-attempt-03\`.
  Attempt 01 stopped at a diagnostic wrapper missing the processor's `n_samples`
  attribute. Attempt 02 completed without recording CUDA peak memory. Both are
  retained; attempt 03 is the complete result.
- CPU tests: 22 passed across `test_audio_runtime.py` and
  `test_runtime_dispatch.py`. Ruff checks passed for the changed Python files.
- The separate Orbitune PID 28376 was observed before and after this short
  inference. It remained running; it was not stopped or modified.

## Decision and next step

Keep `auto` as the default and keep the strict `1e-5` parity gate. The evidence
does not justify hiding the remaining CUDA FP32 backend delta with a wider
tolerance. If one canonical cross-device output is required, the remaining
small mismatch must be resolved or the supported device/precision contract
must be defined explicitly. Audio quality is not established by this runtime
diagnostic; the broader multimodal quality and package-size gates remain open.
