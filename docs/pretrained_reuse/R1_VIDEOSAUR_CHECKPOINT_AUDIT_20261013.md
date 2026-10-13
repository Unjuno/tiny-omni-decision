# VideoSAUR YT-VIS DINOv2 checkpoint audit — 2026-10-13

Status: **checkpoint acquired and safely inspected; runtime not admitted yet**.
Natural-video pretraining makes this a distinct candidate after the unfavorable
CLEVRER StoSAVi transfer diagnostic. No model training or quality evaluation ran.

## Immutable evidence

Official repository: https://github.com/martius-lab/videosaur

Source checkout: `7a9f85d9388fdeb44eb729ce7ada31569a2e77a2` at
`C:/CodexArtifacts/pretrained-reuse-r1/videosaur-source-audit-01/`.
The upstream README links the YT-VIS2021 DINOv2 checkpoint. Code LICENSE is MIT;
README notes inherited files may have their own licenses. The weight repository
has no model card or separate license file in the inspected tree. Therefore
checkpoint redistribution/product admission remains REVIEW, not implicitly MIT.

Weight repository: `andriizadaianchuk/videosaur-ytvis-dinov2-518`

Revision: `29c97e0d591f27affbf09fdfa8c80e8de97e8184`

File: `videosaur_dinov2.ckpt`,374,234,522bytes.

SHA256: `8c6881307ec37b952325239103ad37bb542665ed24257e837c9db560502b468f`.

Streamed from the immutable HF URL and verified exact byte count/hash against
HF LFS metadata. No global dependencies installed or upgraded. Existing data,
weights and runs remain unchanged. No GPU work or cloud charges.

Artifacts:
- source checkout includes `hf-model.json` and `download_audit.py`;
- `C:/CodexArtifacts/pretrained-reuse-r1/videosaur-weight-audit-01/` contains
  checkpoint, `download.json` and `tensor-audit.json` (all shapes/dtypes).

## Actual serialized contents

PyTorch2.6 `get_unsafe_globals_in_checkpoint` returned an empty list;
`torch.load(weights_only=True, map_location='cpu')` succeeded. No allowlist or
unrestricted pickle was used. Top-level entries: epoch,global_step,
pytorch-lightning_version,state_dict,loops,callbacks.

256 state tensors,93,530,201 elements:

| State-key group | Elements |
| --- | ---: |
| initializer.mean + initializer.log_std |256|
| encoder.module |87,957,632|
| processor.module |347,008|
| decoder.module |5,225,305|

These are serialized tensor counts, **not instantiated resident parameter counts**.
The complete model has not loaded. The tensor inventory suggests parameter-budget
feasibility but excludes remaining Omni components and is not a product claim.

## Verified configuration mismatch

Pinned `configs/videosaur/ytvis_dinov2.yml` declares SLOT_DIM64. The actual weights
consistently require128: initializer[1,1,128], encoder output[128,1536], grouper
q/k/v[128,128], predictor qkv[384,128], decoder positions[1,1369,128], decoder
renderer first weight[1024,128]. Output transform[2137,1024] is consistent with
768 reconstruction features +1369 temporal-similarity targets. Backbone config
is DINOv2 ViT-base/14,518x518 inputs,7 slots; seven is a config value rather than
a slot count recoverable from initializer shape[1,1,128].

GitHub file-history API returned only initial code release
`81dd9c1ad8d81458691bbfd3d7884c86867f8862` for this config path. Inspected issue
listing did not resolve the dimensional mismatch. Exact original training config,
including any other differing non-shape settings, is UNKNOWN. Do not present a
manual128-dimension reconstruction as recovered exact training provenance.

The documented environment is Python3.8/Torch1.13/Torchvision0.14 with Lightning,
timm0.9.7,OmegaConf and einops. Current environment has newer Torch; these support
packages are absent globally. Source inference uses unrestricted torch.load;
any local harness must retain safe loading instead. Do not install the entire
old training environment over the working project environment.

## Next bounded admission

A separately recorded inference configuration with width128 can be constructed
from tensor evidence and the remaining pinned source settings. It must strictly
load every inference tensor, disclose unverifiable non-shape settings, and pass
frozen finite-output/save-reload checks before any train-side transfer diagnostic.
Do not silently use strict=False, drop incompatible layers, or quote upstream
segmentation numbers as this project's Decision quality. Neither the checkpoint
nor this audit satisfies the R2 quality gate or the overall Omni objective.
