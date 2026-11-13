# StoSAVi strict CPU runtime admission — 2026-10-13

Status: **PASS_RUNTIME_NOT_QUALITY**. This advances R1 only. No Decision accuracy,
object identity, natural-video transfer or complete Omni parameter claim follows.

## Executed

Source commit of this repository: `2ab8143`. SlotFormer source remains pinned to
`10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa`; dependency nerv v0.1.0 resolved to
`e83ac66c6ce30e1ca3d0a287df9d3699ed9ec499`. No environment upgrade or installation.
Loaded the upstream StoSAVi class/config and unchanged helper modules directly,
without importing unrelated training/dataset entry points. The upstream
`deconv_out_shape` function was AST-extracted unchanged to avoid unrelated imports.
This is an isolated inference import harness, not a full upstream installation.

The CLEVRER StoSAVi checkpoint was loaded with `weights_only=True`, then
`load_state_dict(strict=True)`: zero missing/unexpected keys. All parameters were
frozen, model in eval mode, upstream `testing=True` (skip reconstruction decoder).
No model parameter was modified; complete state comparison after inference passed.

| Measurement | Actual result |
| --- | ---: |
| Resident model parameters, including unused decoder | 1,198,980 |
| Buffer elements | 16,640 |
| Trainable parameters | 0 |
| Saved state bytes | 4,884,196 |
| Input | 1 x 6 x 3 x 64 x 64 |
| Posterior slots | 1 x 6 x 7 x 128 |
| Four forward calls, seconds | .05903 / .05615 / .05120 / .05358 |
| Same-seed replay / strict save-reload | exact / exact |
| Output finiteness | pass |

CPU FP32, four threads; Python3.11.9, Torch2.6.0+cu124,
Torchvision0.21.0+cu124, PyAV18.1.0. Times exclude media decoding, loading and
preprocessing; four calls are not a latency benchmark. Peak RAM was not measured.
No GPU work, training or cloud charge.

## Input and preprocessing correction

Used first six consecutive frames of existing **train** clip `video_10103`,
all before cut_frame459. Media hash checked against the existing manifest;
SHA256 `e9914a811679e99f26246ab0c1f89c582914e37433189bf925f06f5793ca38ed`.
No validation, sealed audit or question labels were loaded.

Attempt01 used PIL bilinear resize before normalization. Upstream dataset code
instead performs ToTensor, Normalize(.5,.5), then tensor Resize. Attempt02 applies
that order with explicit bilinear/antialias=False, matching the historical
Torchvision0.11 tensor default (source inspected), rather than today's changed
default. Attempt01 is preserved as a preprocessing-mismatched runtime smoke;
it is not the admitted extraction path. Historical/current-library numerical
parity has not been measured in an old environment.

StoSAVi `_sample_dist` remains stochastic even in eval mode. Seed17 is reset for
each replay. Seed18 changed slot components (max absolute difference0.309761);
this is an implementation observation, not a quality comparison or seed search.
Any later extraction must define per-video seeds and chunk/state handling before
comparisons. Do not silently replace posterior sampling with its mean.

## Artifacts and reproduction

- Sources/harness: `C:/CodexArtifacts/pretrained-reuse-r1/slotformer-source-audit-01/`
- Original dependency: `C:/CodexArtifacts/pretrained-reuse-r1/slotformer-nerv-v010/`
- Admitted result: `C:/CodexArtifacts/pretrained-reuse-r1/slotformer-stosavi-runtime-02/result.json`
- Preserved earlier result: sibling `slotformer-stosavi-runtime-01/`
- Result contains all imported source hashes, config, environment, timing,
  identity, strict-load/replay checks and saved state/slot artifact hashes.

Executed command:

```powershell
python C:/CodexArtifacts/pretrained-reuse-r1/slotformer-source-audit-01/runtime_audit_tensor.py
```

Harness SHA256 `1e7a04d88a9251553d00e086d9d64d3da9f50a5f249427c98ff8cfb5c19ad4d7`.
Output directory is exclusive-create: replay must use a new directory, never
remove/overwrite prior results. Checkpoint SHA256 remains
`647ed1c177b66cb5507949637c4d61cc0be2f08a015ad82d5f2b456490e89a71`.
Saved state SHA256 `fcc74ec3e00e9c1aac15c89bf4a590b6189a1c27397df98ab527108fe96db8e7`.

## Gate / next action

Checkpoint redistribution rights remain REVIEW as in the checkpoint audit.
The dynamics SlotFormer and ALOE models have not been instantiated. Natural-video
slot quality is unknown. First inspect whether fixed slots/masks retain object
identity on a small train-only diagnostic with explicit preprocessing, seed and
frame policy; do not launch another full validation readout merely because this
small model loads. Any annotation-assisted identity assignment must be labeled
oracle diagnostic and cannot be counted as Decision accuracy. Existing models,
corpora and runs remain unchanged. R2 quality and R3–R8 goals remain unmet.

Follow-up: [fixed train mask diagnostic](R2_STOSAVI_TRAIN_MASK_DIAGNOSTIC_20261013.md) completed; the natural-video evidence does not justify a full Decision fit.
