# Path A image runtime full-validation parity replay

Date: 2026-10-21

Repository source commit: `cf55f6aac383e54afa2579eda60d25fd927fe16d`  
Purpose: verify that the manifest-listed image runtime reproduces the already
saved checkpoint-selection predictions across the full existing 512-image
development validation set. This is packaging/runtime parity only. It is not a
new model evaluation, training run, checkpoint selection, or blind test.

## Inputs and integrity

- Runtime bundle: `path-a-image-videoprism-clevr4-runtime-v0`
- Bundle manifest SHA-256:
  `958e1f8735c33237342d0533f7529a0514fc4f002ca812057b2eff28ac51012e`
- The bundle was read from the separate strict manifest-only integrity copy;
  the original package was not changed.
- Source run report SHA-256:
  `ef130520d00188325f599ae0671505def64b3c5df015c63808dbf815621659ee`
- Baseline checkpoint-selection predictions SHA-256:
  `2336f6efafe4b6d5e1efa4f1257b5f9eb39c03824d7c4a53fda31a920162d30a`
- VideoPrism source checkout HEAD:
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` (runtime-required revision)
- The JAX environment reported Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1,
  NumPy 1.26.4, with only `TFRT_CPU_0` available.
- Validation image identity-list SHA-256:
  `95e8b9e008703bcbfc00d31b9d29706c6e0f54971c2f912405ef015dac8e5c64`

## Execution and result

The existing `scripts/verify_path_a_videoprism_clevr4_runtime.py` was run with
`--images 512`, `CUDA_VISIBLE_DEVICES=''`, and `JAX_PLATFORMS=cpu`. It scored
2,048 decisions from 512 images against the saved baseline predictions.

| Measure | Result |
|---|---:|
| Runtime parity | PASS |
| Prediction mismatches | 0 / 2,048 |
| Maximum absolute probability difference | 0.00000497699 |
| Allowed probability difference | 0.00001 |
| Runtime load | 3.66 s |
| Inference | 547.27 s (1.069 s/image) |
| Peak working set | 4,260,151,296 bytes (~4.26 GB) |
| Training / selection / sealed audit | None / none / not loaded |

The exact run report and all per-decision replay outputs are preserved outside
Git at:

`C:\CodexArtifacts\pretrained-reuse-r2\path-a-image-full-validation-parity-01`

`report.json` SHA-256:
`66d8afaad0b1677acf795fe325d314e6477101c496f74aacb97285e2016871c9`

`runtime-predictions.jsonl` SHA-256:
`edd95439a99f1402c3ff3c62209bc489d34c878d2c49430373982912d6168934`

## Interpretation and limits

The packaged runtime reproduces the saved reference outputs over the complete
512-image checkpoint-selection validation slice within the specified tolerance.
This supports runtime serialization and preprocessing parity for this image
bundle. The same validation data had already been used for checkpoint
selection, so this does not provide independent evidence of image quality,
generalization, or the broader multimodal product target. No weights, media,
corpus, checkpoint, or pre-existing run artifacts were modified.
