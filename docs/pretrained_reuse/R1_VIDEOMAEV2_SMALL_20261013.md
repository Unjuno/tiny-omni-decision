# VideoMAE V2 Small runtime audit — 2026-10-13

Status: strict load and real-video CPU forward passed; Decision quality **unmeasured**. This is OpenGVLab VideoMAE V2, distinct from the previously excluded noncommercial MCG-NJU VideoMAE v1 release. No existing encoder or artifact is replaced.

## Provenance

- Official [model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/MODEL_ZOO.md) links the distilled `vit_s_k710_dl_from_giant.pth` checkpoint.
- [Owner model card](https://huggingface.co/OpenGVLab/VideoMAE2/blob/706cc172d65ebd4dedbee3f9c0183a93df9fa125/README.md) declares Apache-2.0. Preserve attribution and license notices for any later packaging; this audit is not product adoption.
- Source commit: `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`.
- Hub revision: `706cc172d65ebd4dedbee3f9c0183a93df9fa125`.
- Checkpoint: 44,334,609 bytes, SHA-256 `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Executed source file SHA-256: `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`.

## Measurement

Official `vit_small_patch16_224(num_classes=710)` loaded all keys strictly, using `torch.load(weights_only=True)` on CPU. Actual parameter count **22,153,286**, including the 710-way classification head. This is an individual video component, not a complete Omni parameter count. The checkpoint contains FP16 weights; this CPU measurement used FP32.

Frozen/eval model, four CPU threads, PyTorch 2.6.0+cu124 and existing isolated timm 1.0.15 dependencies. No package upgrade. Train video `video_10186`, SHA-256 `6e61d8f411114ef7f052a5bea40bd51b4161274bb0d799cd158658a14b6d4df7`; 16 evenly indexed frames in [0,650], all before the existing exclusive reveal cut 651. RGB full resize 224, divide by 255, mean/std 0.5. This is diagnostic preprocessing, not the official multi-crop benchmark.

Logits shape `[1,710]`, feature shape `[1,384]`, finite outputs, repeated logits bit-identical. Three forwards (logits, features, repeated logits) took **1.1515554 seconds** after loading/decoding in the first successful process. This is not end-to-end latency or a repeated throughput benchmark; GPU performance and VRAM are unmeasured.

The first isolated attempt failed before model construction because the dynamically loaded module was absent from `sys.modules`. Registering it before execution fixed the import. Original failed and successful scripts remain preserved. Legacy timm import and model-registry collision warnings are process-local; no global environment change was made.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-audit-01/` (pinned source, checkpoint, metadata, scripts, runtime JSON and features). Repository runner `scripts/audit_videomaev2_small_runtime.py` adds source/checkpoint hash assertions and requires a new output directory. Reproduction command:

```powershell
python scripts/audit_videomaev2_small_runtime.py --output C:/CodexArtifacts/pretrained-reuse-r1/videomaev2-small-runtime-reproduction-01
```

No training, validation, sealed-audit inspection, quantization or quality selection occurred. The much smaller encoder leaves parameter budget for other modalities, but useful conditional temporal reasoning remains an open requirement. A task-matched comparison needs a fixed protocol and controls before any adoption decision.

Repository-runner reproduction also passed: three forwards 1.4439220 seconds, same shapes and exact within-process repeat. Its output is preserved separately at the command's destination.
