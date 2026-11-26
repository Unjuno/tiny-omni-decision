# Conditional/native candidate: measured component budget

## Result

The nine pinned weight/normalizer files plus13 calibration scalars contain
**195,655,225 tensor elements**, including stored buffers and both normalizers.
This conservative file-level count is below200,000,000 by **4,344,775**.
It is not an exact `named_parameters()` count of a fully assembled model: that
model has not been loaded or qualified. Additional components must fit the margin.

Source commit: `96e00dde2eebb75951c3efb1d66462a72b31e9fa`.
Config: `configs/pretrained_reuse/conditional_omni_budget_v1.json`.
Full result: `C:/CodexArtifacts/pretrained-reuse-r2/conditional-omni-component-budget-01/result.json`.
SHA256: `8a22824775535ff89a91b897622bd72bb72fa214ac2d03d692f735bfd0d54764`.
[Compact receipt with per-file hashes](R2_CONDITIONAL_CANDIDATE_BUDGET_RECEIPT_20261015.json).

Executed on the local Python3.11 environment:

```powershell
python -m scripts.audit_conditional_omni_budget --output C:/CodexArtifacts/pretrained-reuse-r2/conditional-omni-component-budget-01
```

Exit0, all input/code hashes verified before and after accounting. No GPU, model
forward, optimizer, media, transfer or sealed-audit access was used.

## Complete declared inventory

| Component | Stored elements | Tensor bytes |
|---|---:|---:|
| VideoPrism, shared image/video encoder |114,365,184|457,460,736|
| DeBERTa semantic core, including embeddings |70,830,722|141,661,444|
| Whisper encoder |8,208,384|32,833,536|
| EfficientAT, including buffers |1,972,124|7,888,740|
| Actual affine image head |61,480|491,840|
| Full audio temporal head |106,355|425,420|
| Full video temporal head |106,355|425,420|
| Audio normalizer |2,304|9,216|
| Video normalizer |2,304|9,216|
| Calibration scalars, accounted as float64 |13|104|
| **Total** |**195,655,225**|**641,205,672**|

The actual image bridge uses `shared-affine-regularized-01/0.01/final.pt`, a
1536→40 affine head with61,480 parameters. The earlier folded246,016-parameter
head is a different artifact and is not substituted into this count. Both
106,355-element temporal heads are counted in full, including unused branches.
The equal-hash audio/video normalizers are conservatively counted twice.
The calibrated correspondence path contributes6+6+1 positive finite scalars.

The previous195,365,529 figure counted only the four base models' named
parameters. The additional289,696 here accounts for heads, normalization,
calibration and10,885 EfficientAT buffer elements. It does not change any model.
Existing semantic adaptation replaces named parameters in place; its saved delta
is not an additional simultaneous inference model. Packaging must materialize
the selected weights without retaining redundant checkpoint copies.

## What fits and what remains unproved

The nine source files total641,459,157 bytes before any package assembly. Mixed
source dtypes and file overhead make this different from parameter count,
runtime RAM and a quantized distribution. A <=200MB package is **not measured**.
Quantization changes storage precision, not the number of parameters.

Replacing the8,208,384-element Whisper encoder with the37,760,640-parameter full
Whisper adds29,552,256 parameters. On this accounting basis that would reach
225,207,481, exceeding the cap. Encoder-only speech semantics must therefore
demonstrate the required quality, or allocation/architecture must change.
No sensor or extra Japanese-language encoder is hidden outside the budget.

This is the current research component inventory. It proves neither four-modal
95–98% quality nor general temporal reasoning, novel conditional-language
transfer, arbitrary speech understanding, complete integration or runtime cost.
The fixed image taxonomy and narrow temporal heads are not general Omni quality.
Any future connection layers, language support or replacement head must be
recounted and evaluated on the same required-task contract.

Four CPU tests verify numpy/safetensors/torch counts, buffer inclusion, corrupted
offsets, optimizer-container rejection and a boundary crossing above200M.
Read-only review confirmed provenance and scope with no blocking finding.

## Strategy consequence

There is a plausible numerical path under200M, with only4.34M conservative
headroom. Keep the existing modular baseline for the next controlled condition
comparison. Do not add a larger language/audio model or begin compression to
compensate for unproven quality. Advance only after independent condition,
native perception, temporal/joint and retention evidence agree for one candidate.
