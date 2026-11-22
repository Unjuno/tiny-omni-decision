# Path A encoder-only four-route replay — 2026-11-22

## Result

Replayed the same fixed Video, Image, Text and Audio development-validation
requests through the encoder-only assembly using the existing CPU dispatcher
benchmark. This is a runtime-preservation check against saved historical
predictions, not a fresh quality evaluation. Target labels were not used for
scoring; no weights were trained and no sealed-audit data was loaded.

The encoder-only assembly reproduced the previous full-Whisper assembly's
comparison deltas exactly for all four routes and both first/warm requests.
Typed JSON passed on all routes. Video, Image and Text parity passed. Audio
parity remains failed at the already-known CPU FP32 vs saved CUDA BF16 delta of
`6.1774253845e-4`; the compact encoder-only storage did not change that delta.
Therefore the aggregate runtime gate remains failed until the pre-existing
Audio precision mismatch is addressed under a separately justified change.

| Route | Full assembly first / warm delta | Encoder-only first / warm delta | Parity | Typed JSON |
|---|---:|---:|---|---|
| Video | 0 / 1.4901161e-7 | 0 / 1.4901161e-7 | pass | pass |
| Image | 1.2805685e-8 / 2.3841858e-7 | 1.2805685e-8 / 2.3841858e-7 | pass | pass |
| Text | 5.6624413e-7 / 5.6624413e-7 | 5.6624413e-7 / 5.6624413e-7 | pass | pass |
| Audio | 6.1774254e-4 / 6.1774254e-4 | 6.1774254e-4 / 6.1774254e-4 | known fail | pass |

First-request dispatch times on this CPU run were 34.58 s Video, 2.20 s Image,
0.08 s Text and 0.40 s Audio; warm calls were 0.02 s, 0.01 s, 0.09 s and
0.004 s, respectively. Total sequential runtime was 41.32 s. These are one
request per route, not latency benchmarks. Combined post-load worker RSS was
1,645,756,416 bytes (Torch 830,885,888; VideoPrism 814,870,528); this is not a
matched memory comparison against the full assembly.

## Provenance

- Source commit: `78d40eb5d7038227b5755ba7e34ec9c9e7de3d5d`.
- Encoder-only assembly manifest SHA-256:
  `616b9b61e983d875a6a29e71d568f1ba0a6aeba6705da61ee66a88b4be43b3fd`.
- Encoder-only run report:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-encoder-only-four-route-cpu-replay-20261122-attempt-01\dispatcher-benchmark.json`.
- Run report SHA-256:
  `e8c164e600c1c6a12d67e3a5c519cbab928fbecd9ea283bc882090dd46768490`.
- Full-assembly comparison report:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-shared-four-route-cpu-replay-20261122-attempt-01\dispatcher-benchmark.json`.
- Full-assembly report SHA-256:
  `1e4786d0eaead2944f91e83cff06c10f1059bf98a23f110a430e45347dd051c7`.
- Environment: Windows, Python 3.11.9 / Torch 2.6.0+cu124 CPU dispatcher;
  VideoPrism environment Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1,
  pinned VideoPrism revision `be719a406d563b66f0ac969e7c94bab8e997c81a`.
- VideoPrism source was the local pinned checkout at commit
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Report confirms `sealed_audit_loaded: false` and `training_performed: false`.

Reproduction from the repository root (outputs must use a fresh attempt path):

```powershell
$env:PYTHONPATH = "src"
$env:CUDA_VISIBLE_DEVICES = ""
python scripts/benchmark_path_a_dispatcher.py `
  --assembly "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01" `
  --videoprism-python "C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe" `
  --videoprism-source "C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source" `
  --artifact-root "D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2" `
  --output-dir "<new unique attempt directory>" `
  --device cpu --audio-encoder-precision fp32
```

This adds confidence that the encoder-only storage variant preserves all four
fixed routes. It does not resolve the Audio precision discrepancy, improve
task quality, establish natural multimodal fusion, or close the overall model
size and quality targets.
