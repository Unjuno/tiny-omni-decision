# Small aligned image–text checkpoint audit

This fills an R1 inventory gap while a separate text-training job remains active. No GPU inference or training was added. Existing model/data artifacts were preserved.

## TinyCLIP: actual released checkpoint

- Repo `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, immutable revision `07a4b0bc751cb64fecd2b661c048c1dd98d69444`.
- Weight file332,443,148bytes, SHA256 `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`, matching Hub LFS metadata.
- Loaded actual parameters **83,101,185**, including text embeddings and both projections; FP32 parameter storage332,404,740bytes. The model name39M+19M is not the complete deployment count.

| Component | Parameters |
| --- | ---: |
| logit_scale | 1 |
| text_model | 44251648 |
| vision_model | 38325248 |
| visual_projection | 262144 |
| text_projection | 262144 |

Hub tensor metadata reports83,101,184floating elements plus274integer elements; actual model counting includes the scalar logit scale. Use actual resident model parameters, not metadata arithmetic or names, for the product gate.

Two solid-color images and two short captions were processed on CPU with2threads. Both image/text embeddings have shape[2,512]; repeated logits match exactly and are finite. No missing/unexpected/mismatched load keys. The independently saved reproduction also matched the first audit logits/count. This is loading/preprocessing evidence, not meaningful image QA accuracy or calibration. Runtime/package versions and file hashes are in JSON.

TinyCLIP plus existing Whisper-tiny37,760,640 and Decision-text22,713,601 would total143,575,426parameters before any fusion/readout. This is arithmetic feasibility only, not an integrated model. Retaining VideoPrism114,365,184 as well would exceed200M. Framewise image encoding does not provide native temporal reasoning, so a video claim requires new measured evidence.

The [publisher model card](https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M/blob/07a4b0bc751cb64fecd2b661c048c1dd98d69444/README.md) declares MIT and identifies the Microsoft TinyCLIP implementation. This audit uses released weights, not YFCC media ingestion. Downstream media remain subject to dataset gates.

## MobileCLIP and MobileViCLIP: do not treat code MIT as weight permission

The [Apple model license](https://github.com/apple-aiml-research/ml-mobileclip/blob/48faa0fea4b08d74188b3841771aca6ff2c92852/LICENSE_MODELS) restricts the weights/derivatives to research purposes and excludes product development. Therefore these weights are not an ALLOW product path under the current project gate. No MobileCLIP weights were downloaded.

[MobileViCLIP documentation](https://github.com/MCG-NJU/MobileViCLIP/blob/3706de2e69d30be05c08f8fb8a4ba0619f76ea43/README.md) says it starts from pretrained MobileCLIP. Its repository MIT file alone does not resolve inherited weight rights. Keep it REVIEW; no weight download or integration.

## Next evidence needed

A task-matched, question-conditioned image comparison with fixed candidates and image-shuffle/text-only controls would test whether pretrained image–text alignment helps beyond the unrelated frozen-tower readout. Select exact tasks before inference; do not equate caption similarity or a tiny color smoke with general Decision reasoning. Audio, video, fusion and95% per-modality remain unproved.

Reproduce the CPU audit:

```powershell
python -m scripts.audit_tinyclip_checkpoint --model C:/CodexArtifacts/pretrained-reuse-r1/tinyclip-39m19m-audit-01/model --output <new-json-path>
```
