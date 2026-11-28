# Physion++ VideoPrism CPU feature smoke — 2026-11-28

## Result

The pinned VideoPrism-B encoder successfully loaded in the dedicated local
Python environment and extracted one frozen feature from the first
deterministic row in the existing TRAIN manifest. `JAX_PLATFORMS=cpu` exposed
only `TFRT_CPU_0`; no GPU was used. The run verified the pinned VideoPrism
checkpoint SHA, upstream source commit, train-video SHA, manifest frame count,
and finite `(768,)` pooled feature output.

| Item | Measurement |
|---|---|
| Train record | `physionpp/readout:readout_data_v1/deform_clothhang_pp/deform_clothhang-zdloc=2-target=sphere#0000` |
| Label (record metadata only) | No |
| Video SHA-256 | `3b5374cd7297302b6829d50283e49bbd5a3eb8976a60ffdf25d09f806ecf70cd` |
| Frames | 8 of 724; indices `0, 103, 207, 310, 413, 516, 620, 723` |
| Pooled feature | 768 finite float32 values |
| Elapsed including model load and first forward | 9.06 s |
| Process RSS before model load / after forward | 367,177,728 B / 1,131,368,448 B |
| Runtime | Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1, PyAV 18.1.0, NumPy 1.26.4 |
| Checkpoint | `google/videoprism-base-f16r288`, revision `be719a406d563b66f0ac969e7c94bab8e997c81a`, SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` |
| VideoPrism source | `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` |

The elapsed time includes model initialization/JIT and is not a steady-state
per-video throughput estimate. This is runtime/input-pipeline evidence only;
there is no fitted readout, validation score, or quality claim. It does not
close R2 or the human observability review gate.

## Reproduction and artifacts

```powershell
$env:JAX_PLATFORMS = 'cpu'
C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe `
  scripts/smoke_physionpp_videoprism_runtime.py `
  --train C:\CodexArtifacts\tqpp\fullclip-v1\train.jsonl `
  --corpus-root C:\CodexArtifacts\tqpp\fullclip-v1 `
  --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz `
  --source-root C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source `
  --output C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-videoprism-cpu-smoke-20261128-attempt-01
```

- Smoke script SHA-256: `7557f63e7ebb8db42e8747cae599d1c3f8f2c0136f57f699f2271f162bb276b7`
- Focused tests: `tests/test_smoke_physionpp_videoprism_runtime.py` — 5 passed
- Feature NPY SHA-256: `dd7e1018a386b24667293330ea793b3d86ff380db4411a092d82d28a19317a24`
- Machine report SHA-256: `b45baf332fa7400e9c4025988ea4822e704cb7bf42290ec6e4c921e942394ca6`
- Output directory: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-videoprism-cpu-smoke-20261128-attempt-01`
