# PE-Core T16 CPU runtime revalidation — 2026-10-25

## Result and scope

An isolated CPU-only forward was rerun from the locally pinned converted
checkpoint while the unrelated Teacher training process retained the GPU. The
runtime returned finite image and text embeddings with shapes `[1,512]` and
`[2,512]`; two encoder calls plus model construction took 1.191 seconds with
two CPU threads. CUDA was not initialized. The exact instantiated model has
69,546,113 parameters. No task dataset was accessed in this revalidation.

This establishes a reproducible CPU inference path for the pinned
`OpenCLIP 3.3.0` conversion. It does not establish full tensor-by-tensor parity
with Meta's original checkpoint, compatibility with the project dispatcher,
or product rights. The checkpoint is listed as Apache-2.0 on Hugging Face,
while the official `perception_models` package at the pinned source revision
declares FAIR Noncommercial Research terms. Keep the candidate's license status
at `REVIEW` pending an explicit rights determination.

## Pinned evidence

- Original checkpoint: `facebook/PE-Core-T16-384@8b4722741a516c18e7d311f158f13c600fea6c78`
- Original file: 278,284,008 bytes; SHA-256
  `868f89f066b7c1ae51add02d2cb7b925fd051076282773f2ec3c71359fbdaf3b`
- Converted checkpoint: `timm/PE-Core-T-16-384@7fe539ed578ac49a1c2b4f946e4b0747704c825a`
- Converted file: 278,219,132 bytes; SHA-256
  `477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`
- CPU revalidation receipt:
  `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-05-runtime-revalidation\result.json`
- Receipt SHA-256:
  `65da96409b5414ce02d0373846bc980a3d3d8cf9b7f709e5f63613503a367c4a`
- Environment: Python 3.11, PyTorch 2.6.0+cu124, OpenCLIP 3.3.0,
  timm 1.0.25; local packages and weights from the prior isolated runtime
  attempt; offline Hub mode; `CUDA_VISIBLE_DEVICES=-1`.

The original checkpoint's CPU-safe state-dict audit also remains present and
matches its saved receipt: 313 finite float32 tensors, 69,545,601 elements,
with state-dict audit JSON SHA-256
`1d01e6b849e64b83cec595550a356b7e58d4727681b1e40bbe56f165c96777ec`.
The runtime count differs by 512 because OpenCLIP constructs a zero-initialized
`visual.trunk.head.bias` absent from the original state dict.

## Existing task evidence; development only

The independent `codex/r1-pe-core-audit-20261021` branch recorded fixed
synthetic CLEVR-4 comparisons. Its result artifacts were rehashed locally on
2026-10-25 and matched the report receipts:

- Direct zero-shot similarity: macro Accuracy 42.81% on 1,595 previously
  observed reporting images.
- A separately trained 131,329-parameter candidate scorer over frozen features:
  macro Accuracy 79.95%; Count Accuracy 29.28%, compared with 88.06% macro
  Accuracy for the historical Path A predictions on those same reporting IDs.
- The trained scorer's validation NLL improved through all eight scheduled
  epochs; the run selected epoch 8. This one-seed, synthetic development result
  does not establish a clean plateau, blind quality, or broad transfer.

Full preregistration, metrics, and caveats remain on the independent branch:
[zero-shot protocol](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r1-pe-core-audit-20261021/docs/pretrained_reuse/R2_PE_CORE_CLEVR4_ZERO_SHOT_PROTOCOL_20261021.md),
[zero-shot results](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r1-pe-core-audit-20261021/docs/pretrained_reuse/R2_PE_CORE_CLEVR4_ZERO_SHOT_RESULTS_20261021.md),
[readout protocol](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r1-pe-core-audit-20261021/docs/pretrained_reuse/R2_PE_CORE_CLEVR4_READOUT_PROTOCOL_20261021.md), and
[readout results](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r1-pe-core-audit-20261021/docs/pretrained_reuse/R2_PE_CORE_CLEVR4_READOUT_RESULTS_20261021.md).
The branch is preserved; its worktree and artifacts were not modified.

## Reproduction

The revalidation used only the locally cached, hash-pinned converted checkpoint
and fixed synthetic smoke inputs. It ran in a separate Python process with the
attempt-02 package target and Hugging Face cache, offline mode, two CPU threads,
and CUDA disabled. The complete serialized measurements are in the external
receipt listed above; no weights, media, or run outputs are included in Git.

Primary references: [original PE-Core checkpoint card](https://huggingface.co/facebook/PE-Core-T16-384),
[converted OpenCLIP checkpoint card](https://huggingface.co/timm/PE-Core-T-16-384),
[pinned PE implementation license](https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/LICENSE.PE), and
[pinned implementation package metadata](https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/setup.py).
