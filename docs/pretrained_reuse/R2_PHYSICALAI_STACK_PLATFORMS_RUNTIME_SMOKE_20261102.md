# PhysicalAI stack-platforms frozen-feature runtime smoke — 2026-11-02

## Scope

This is a feature-path smoke on three frozen train and three frozen validation
episodes from split v3. It processes four windows per episode (24 queries),
verifies Parquet timestamp/frame agreement, and uses the exact DINOv2-base
checkpoint planned for the fixed pilot. It is not a readout fit or validation
quality evaluation; no validation targets or metrics were inspected.

Episodes: train `8510, 3736, 12151`; validation `5491, 1218, 9434`.

## Exact runtime and parity

- DINOv2-base revision: `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- `model.safetensors`: 346,345,912 bytes, SHA-256
  `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`.
- `config.json` SHA-256:
  `f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70`.
- `preprocessor_config.json` SHA-256:
  `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`.
- Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0,
  PyArrow 25.0.1, Pillow 10.4.0, NumPy 2.4.6; FP32. CUDA matmul TF32 was
  disabled; cuDNN TF32 was enabled.
- Device: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB. A contemporaneous
  separate Orbitune process remained live; no process or sentinel was changed.

CUDA batch 4 processed the 24 queries in 19.37 seconds total, including model
loading and artifact hashing; measured inference was 15.67 seconds, frame
decode 1.97 seconds, and peak PyTorch CUDA allocation 402,263,040 bytes. The
GPU was heavily shared, so this is only a feasibility observation, not an
isolated throughput benchmark. CUDA batch 1 and batch 8 also completed; peak
allocations were 367,797,248 and 448,883,712 bytes respectively. The smoke did
not show a consistent speed gain for batch 8 under the changing background GPU
load, so the planned full extraction uses batch 4.

CPU FP32 on the same episodes took 38.86 seconds of DINO inference and 1.52
seconds of decode. CPU and CUDA outputs matched exactly for state, actions,
episode IDs, windows, and timestamps. Maximum absolute feature differences
were `3.05e-5` for eight-frame means and `4.22e-5` for last-frame features.
CUDA batch 1 vs batch 4 differed by at most `3.76e-6` / `1.17e-5`; batch 1 vs
batch 8 differed by at most `4.29e-6` / `1.34e-5` for those features.

## Artifacts and implementation checks

The successful CUDA batch-4 smoke is stored outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-pilot-cuda-batch4-smoke-20261102-attempt-02/`.
`features.npz` SHA-256 is
`1abcaa5610559fe8d61b5628f400984a189239e2f0f33da13c157c4f9012dd37`;
`feature-extraction.json` SHA-256 is
`8f41269cc26aa40bafa4f4670fea780dbd9cf264d745e0af4980ef914811dd35`.
The receipt parser now reads the nested selection schema; direct-script import
works on Windows. The evaluator smoke on synthetic feature arrays confirms
dynamic episode/query counts, train metrics, per-class metrics, and exact
checkpoint reload probability parity. Synthetic numbers are software checks,
not dataset results.

The 700/250 feature extraction and fixed one-shot readout have not yet run.
Keep the source unadmitted and do not inspect or use reserved episodes.
