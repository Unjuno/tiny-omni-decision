# PE-Core T16 pinned checkpoint and CPU audit — 2026-10-21

## Scope and decision

This is a checkpoint integrity and CPU-safe deserialization audit only. It is not a model forward test, task evaluation, training run, or deployment approval. The pinned public checkpoint is now verified locally, but runtime/product eligibility remains **REVIEW** because the official `perception_models` package declares FAIR Noncommercial Research terms. The model card's Apache-2.0 checkpoint notice does not by itself relicense that implementation.

## Pinned source and artifact

- Model: `facebook/PE-Core-T16-384`
- Hub revision: `8b4722741a516c18e7d311f158f13c600fea6c78`
- Hub model card: https://huggingface.co/facebook/PE-Core-T16-384/tree/8b4722741a516c18e7d311f158f13c600fea6c78
- Card metadata: `license: apache-2.0`; upstream describes PE Core as an image/video vision-language contrastive encoder.
- Checkpoint: `PE-Core-T16-384.pt`
- Expected/observed bytes: `278284008`
- Official Hub LFS SHA-256 / local SHA-256: `868f89f066b7c1ae51add02d2cb7b925fd051076282773f2ec3c71359fbdaf3b`
- Local checkpoint: `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-01\PE-Core-T16-384.pt`
- Official implementation source revision: `facebookresearch/perception_models@3e352cca660658d4b5c90f42a7808b11469e4c66`
- `setup.py` labels package license `FAIR Noncommercial Research License`; `LICENSE.PLM` has the same noncommercial research terms. `LICENSE.PE` is Apache 2.0. Keep checkpoint and implementation grants distinct pending legal/product review.

## Safe-load result

Loaded in a separate Python 3.11.9 process with PyTorch `2.6.0+cu124`, using `torch.load(weights_only=True, map_location="cpu")` and `CUDA_VISIBLE_DEVICES=-1`. No repository model code was imported. CUDA was not initialized before or after loading.

- Result: `PASS_CPU_SAFE_STATE_DICT_AUDIT`
- Tensors: 313; dtype: all `torch.float32`; non-finite tensors: 0
- Total state-dict elements: 69,545,601
- Top-level tensor elements: text transformer 37,828,608; token embedding 25,296,896; visual tower 6,140,544; text projection 262,144; positional embedding 16,384; final norm 1,024; logit scale 1.
- Tensor payload bytes: 278,182,404; serialized checkpoint bytes are 278,284,008.
- This observed state dict includes image and text components. It does **not** establish that all tensors are runtime parameters or provide a forward-pass validation. The earlier inventory estimate of 50,000,000 is withdrawn: the complete checkpoint state dict has 69,545,601 elements, but exact instantiated model parameter count remains UNKNOWN until a compliant runtime can instantiate it.

External audit artifacts (not committed):

- State-dict audit JSON SHA-256: `1d01e6b849e64b83cec595550a356b7e58d4727681b1e40bbe56f165c96777ec`
- Audit script SHA-256: `10794af0b740492d202b653da292a55987d3562673bf2002285f5cfc6ebff800`
- Script and JSON directory: `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-01`

## Runtime and quality gates

At the initial audit, no forward pass was attempted. The system Python lacked `timm` and `open_clip`, and stock Transformers compatibility had not been established. The results below supersede that initial runtime status. Image/video/text task quality remains unmeasured, and this encoder does not provide audio. Runtime feasibility is now CPU-smoke-tested; product suitability and end-task quality are still open.

### Follow-up: independent Transformers compatibility screen

A read-only CPU inspection was performed after this report was drafted. It confirms that the checkpoint has an OpenCLIP-like ViT/text-transformer layout, but **does not directly fit stock `transformers.CLIPModel`**:

- Text tower dimensions are 12 layers, width 512, 8 heads, context length 32, with a 49,408-token embedding table.
- Vision tower dimensions are 12 layers, width 192, 3 heads, 384 px input and 16 px patches. The state dict has a learned class embedding and a 577-entry position table (one class token plus a 24×24 patch grid). The pinned config sets `use_cls_token=True`, while the pinned README says only the L and B models additionally have a class token; this T16 documentation discrepancy must be resolved against a permitted reference before claiming parity.
- The checkpoint also includes `visual.attn_pool.*`: an 8-head learned pooling block with a learned probe, attention projections, layer norm, MLP and output projection. The stock Transformers CLIP vision model inspected here exposes the standard class-token pooler, not this PE pooling block.
- `visual.ln_post.*` and `visual.attn_pool.layernorm.*` are distinct learned norms. A faithful independent adapter must establish and reproduce their ordering, and implement PE attention pooling; substituting the stock CLIP pooled output would not be an exact checkpoint runtime.
- The text and vision transformer block tensors use fused QKV weights (`attn.in_proj_*`), while Transformers stores separate Q/K/V projections. This can be split deterministically, but the forward parity still needs verification. The pinned model repository has no usable model config to settle activation and preprocessing details by itself.

The initial assumption that no independent runtime path was available was too strong. Official OpenCLIP now includes `PE-Core-T-16-384`; this support is visible in its upstream package, and the corresponding pinned `timm/PE-Core-T-16-384` Hub repository publishes Apache-2.0 converted weights. The local system Python was not changed: OpenCLIP 3.3.0, timm 1.0.25, ftfy 6.3.1 and wcwidth 0.2.13 were installed with `pip --target` into the attempt-02 artifact directory and exposed only to a separate process.

Using `open_clip.create_model_and_transforms("PE-Core-T-16-384", pretrained=<local pinned safetensors>, device="cpu")`, the converted Hub weights loaded and image/text encoding ran. The pinned Hub revision is `timm/PE-Core-T-16-384@7fe539ed578ac49a1c2b4f946e4b0747704c825a`; `open_clip_model.safetensors` is 278,219,132 bytes with SHA-256 `477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`. Selected tensors match the original local Meta checkpoint exactly (patch convolution, class embedding, text token and position embeddings, text projection, first visual fused QKV weight/bias, and attention-pool probe). This establishes that the tested runtime used the intended pretrained weights; it is not a tensor-by-tensor parity proof.

Smoke input was one all-zero 384×384 RGB tensor and two fixed text strings (`a red car`, `a quiet room`). Returned image/text embeddings had shapes `[1,512]` and `[2,512]`, were finite, and CUDA was not initialized. Elapsed time from before model construction through both encoders was 0.968 s with the already cached artifact; it excludes package installation, process start and Hub download. Runtime parameter count was 69,546,113. The original checkpoint state dict contains 69,545,601 elements; the 512-element difference is OpenCLIP's zero-initialized `visual.trunk.head.bias`, which has no corresponding original tensor. Do not claim exact tensor parity until the entire mapping is audited.

An attempted strict direct load of the original Meta `.pt` state dict into OpenCLIP failed because its key layout differs (original `visual.transformer.resblocks.*` versus the timm-backed `visual.trunk.*` model). The successful run used the converted, pinned Hub safetensors instead. The configuration and README still differ on T16's class-token description (the config and local tensors both indicate a class token). This documentation discrepancy, the 512 added zero bias, and lack of full numerical parity remain recorded limitations.

`runtime_supported` is now **CPU_SMOKE_PASS** for the OpenCLIP 3.3.0 profile; task quality, calibrated decision performance, GPU speed, saved/reloaded parity and integration into the final runtime remain unmeasured. The model is an image/text contrastive encoder, not a Decision head. No task dataset was opened during this runtime smoke.

Pinned architecture references: [official PE T16 config](https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder/config.py) and [official PE Core model description](https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/apps/pe/README.md). These references are documentation only; their availability does not change the separate implementation-license review above.

This follow-up used CPU-only loading and inference in a separate process. It did not import or execute `perception_models`, access task datasets, train, use the GPU, or modify either checkpoint. The project Python environment was not modified; temporary packages and the converted weights are isolated in `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-02-runtime`.

## Safety and scope

No GPU was initialized or used; no training, dataset access, checkpoint mutation, or sealed-audit access occurred. The active unrelated GPU job remained untouched. The existing PE checkpoint and official implementation checkout were read-only; only this branch's inventory/report files were changed.
