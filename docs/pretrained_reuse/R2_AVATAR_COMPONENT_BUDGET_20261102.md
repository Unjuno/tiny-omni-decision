# AVATAR cross-source component budget addendum — 2026-11-02

This is read-only arithmetic for adding the selected AVATAR joint audio/video
readout to the existing Path A v4 task-component union. It does not modify the
v4 bundle or assert that the two packages have been integrated or run together.

## Hash-bound inputs

The base inventory is the existing v4 package manifest with SHA-256
`79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`. It
declares 170,499,716 unique parameters and 687,254,772 package-file bytes,
excluding the package manifest. Its VideoPrism file is
`components/image/videoprism-base-f16r288.npz` (457,475,296 bytes,
SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`),
which exactly matches the VideoPrism checkpoint used for the AVATAR readout.
Therefore the union can count that frozen video encoder once.

| Added AV component | Parameters | File bytes | SHA-256 |
|---|---:|---:|---|
| EfficientAT DyMN-0.4 | 1,961,239 | 8,086,973 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` |
| AVATAR option scorer | 73,857 | 297,472 | `49cc9dd20fabf045960482e97d56bb5bec6d626cfc815b3834468700fd26d05e` |
| Train-derived normalization | 0 | 61,179 | `1579fd6cfc1689ac70d1bc89d1adcd842afefe4be53ad20bd4d1c8ab9a46fdc5` |

## Arithmetic only

| Candidate union | Parameters | Payload bytes excluding root manifest |
|---|---:|---:|
| Existing Path A v4 task union | 170,499,716 | 687,254,772 |
| AVATAR additions with shared VideoPrism | 2,035,096 | 8,445,624 |
| **Arithmetic union** | **172,534,812** | **695,700,396** |

The union is within the provisional 100–200M pre-quantization parameter range.
Its unquantized payload exceeds the provisional 200MB package target by
495,700,396 bytes. No quantization was attempted because the modality/task
quality gate has not passed.

## Not demonstrated

- The existing package manifest still says `supports_joint_audio_video=false`.
- The AVATAR runtime accepts frozen feature vectors; raw-media re-extraction is
  blocked by the current Windows Code Integrity policy, documented in the
  [runtime report](R2_AVATAR_FEATURE_RUNTIME_20261102.md).
- No composite bundle, shared process/residency, complete modality dispatcher,
  simultaneous load, calibrated joint probability, or end-to-end latency was
  tested.
- The task-specific validation result is selection-affected and was not
  rerun for this accounting. The runtime smoke used TRAIN-only data and is not
  quality evidence.

This shows only that the compact AV head can fit numerically alongside the
existing Path A component union if the identical VideoPrism weights are shared.
It does not close R2-M, R7, or the product release gates.
