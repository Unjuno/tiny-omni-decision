# VideoMAE V2 real-video CUDA runtime audit — 2026-11-10

## Result

**PASS for pinned-checkpoint CUDA runtime feasibility; no quality claim.** The
released OpenGVLab VideoMAE V2 Small weights loaded strictly and processed 16
hash-verified frames from one existing Perception Test **training** video on
the local RTX 3080 Laptop GPU. Outputs were finite and repeated logits were
bitwise identical. No training, validation, or sealed-audit data were used.

This supplements the prior CPU real-video audit and synthetic CUDA smoke. It
does not establish VideoMAE decision quality, temporal reasoning quality, or
end-to-end latency including decode/model startup.

## Pinned source and rights

- Owner model card: [OpenGVLab/VideoMAE2 at revision
  `706cc172d65ebd4dedbee3f9c0183a93df9fa125`](https://huggingface.co/OpenGVLab/VideoMAE2/tree/706cc172d65ebd4dedbee3f9c0183a93df9fa125)
  declares Apache-2.0.
- Upstream source commit:
  `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`.
- Local model source SHA-256:
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`.
- `vit_s_k710_dl_from_giant.pth`: 44,334,609 bytes; SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Strict-loaded parameter count: 22,153,286, including its 710-class head.

## Fixed input and measurement

- Video: `video_10186`, Perception Test train split, SHA-256
  `6e61d8f411114ef7f052a5bea40bd51b4161274bb0d799cd158658a14b6d4df7`.
- Frames: indices `[0, 43, 86, 130, 173, 216, 260, 303, 346, 390, 433,
  476, 520, 563, 606, 650]`, before the existing exclusive reveal cut at 651.
- Diagnostic preprocessing: RGB resize to 224, `/255`, mean/std 0.5. This is
  not the official multi-crop benchmark preprocessing.
- Output: logits `[1,710]`, features `[1,384]`; both finite; repeat logits
  bitwise identical.
- Timed forward for logits plus `forward_features`: **73.76 ms**, after one
  warm-up forward and excluding model load, video decode, and input transfer.
- Peak CUDA allocated: **240,692,736 bytes**; peak reserved:
  **274,726,912 bytes**. These include model and input residency.
- Hardware/software: NVIDIA GeForce RTX 3080 Laptop GPU (16 GiB); PyTorch
  2.6.0+cu124, CUDA 12.4, PyAV 18.1.0, timm 1.0.15, FP32.
- No cloud compute or model/environment upgrade. The pinned timm dependency
  emits legacy import/registry warnings; they were non-fatal and confined to
  this process.

## Reproduction and artifacts

Run from the repository checkout, using the already-installed pinned local
dependency directory and a new output directory:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='C:\CodexArtifacts\pretrained-reuse-r1\python-deps'
python scripts\audit_videomaev2_real_video_cuda.py --output D:\CodexArtifacts\pretrained-reuse-r1\videomaev2-real-video-cuda-20261110-attempt-02
```

- Source commit: `e07e049ff4387484fb754e0fff963631f3cc7c85`.
- Runner SHA-256: `c7ef20b645e5cb88367f4daa2588b13ae9783207363571df59e438dbd16bf80d`.
- Runtime receipt:
  `D:\CodexArtifacts\pretrained-reuse-r1\videomaev2-real-video-cuda-20261110-attempt-02\runtime.json`,
  SHA-256 `96c7dc4c82a9c1830032d309a82b0a9e8f51172cb011ba743a62aa793494e02b`.
- Cached feature:
  `D:\CodexArtifacts\pretrained-reuse-r1\videomaev2-real-video-cuda-20261110-attempt-02\features.npy`,
  SHA-256 `8c2ebb45a48834e0cddb9872c8130603696919ace5d25c69bc606d4eec07adae`.
- The earlier successful pre-format smoke is preserved separately as
  `attempt-01`; the report uses the formatted runner's `attempt-02` result.

## Interpretation and next gate

The small VideoMAE checkpoint is a legally usable, locally executable video
feature source and fits comfortably in the measured GPU memory envelope. This
supports using it in a Path A task-specific video probe; it does not replace
VideoPrism, establish superior quality, or close the overall R1/R2/product
gates. Before any quality comparison, freeze an admitted scene-grouped video
Decision task, use train-only fitting and a separate validation selector, and
compare VideoMAE against the existing frozen VideoPrism readout on identical
examples and preprocessing. Do not use the train-video classification output
as a Decision accuracy result.

No validation/audit labels, existing model weights, datasets, checkpoints,
Teacher artifacts, or prior runtime outputs were modified. The new code and
this report are kept on a separate branch; there is no product promotion.
