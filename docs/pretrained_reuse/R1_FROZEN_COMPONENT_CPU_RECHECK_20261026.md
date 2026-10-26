# R1 frozen component runtime recheck — 2026-10-26

Cross-branch reconciliation of an already executed CPU audit. Fresh-process
result files and local model artifacts were read and SHA-256 checked; no model
was trained and no CUDA device was used for these component forwards.

## Revalidated components

| Component | Exact source | Parameters | Weight bytes / SHA-256 | Runtime result |
|---|---|---:|---|---|
| VideoPrism-B | `google-deepmind/videoprism@c6eba0ece3dd49a24ffc44582966997a4a59f1d8` | 114,365,184 | 457,475,296 / `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | 8-frame CPU input `[1,8,288,288,3]`; finite output `[1,2048,768]`; output SHA matched the prior audit |
| Ruri v3-30m | `cl-nagoya/ruri-v3-30m@24899e5de370b56d179604a007c0d727bf144504` | 36,705,536 | 146,828,152 / `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832` | Japanese/English plumbing text; finite hidden `[2,32,256]` and pooled `[2,256]` |
| EfficientAT DyMN-0.4 AudioSet | `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806` | 1,961,239 | 8,086,973 / `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | Strict state load; generated 440 Hz tone produced finite logits `[1,527]` and pooled `[1,384]` |

The three towers sum to **153,031,959 parameters** and **612,390,421 weight
bytes**, excluding preprocessing, tokenizers, and task-specific Decision
readouts. This subset fits the provisional 200M parameter engineering budget;
the raw weights alone exceed the provisional 200 MB package target. This is a
capacity/runtime accounting result, not evidence that the subset is the final
architecture or that compression can meet the size target.

## License scope and runtime environment

The rechecked VideoPrism source revision is Apache-2.0. The pinned Ruri card
records Apache-2.0. EfficientAT source code is MIT, and the checkpoint owner
explicitly permits commercial use in [upstream issue #44](https://github.com/fschmid56/EfficientAT/issues/44).
These are artifact-scoped checks, not a full legal review.

- VideoPrism: Python 3.11.16, JAX 0.6.2, Flax 0.11.1, TensorFlow CPU 2.18.1,
  `TFRT_CPU_0`; 8.12 s wall time and 3,912 MB peak process working set.
- Ruri / EfficientAT: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  CPU only; EfficientAT fixture run 1.25 s.
- No package was upgraded, no GPU or cloud compute was used, and no task quality
  evaluation occurred. The English Ruri string and generated tone are plumbing
  fixtures only.

## Artifact verification

The source run was performed against project commit
`2538c5a6f3702e8535c339984a43af9b9d293fed`. The local read-only receipts were
verified against their files:

| Artifact | SHA-256 |
|---|---|
| VideoPrism result JSON | `09c2c3d34f7a2042499a8c598f42dbb5aadf1ff5ce5762592d2d5fb2a951ee53` |
| Ruri + EfficientAT result JSON | `582812a0e017c2d6d95300d5e95ff26d26d508f15aa6461f94cf107907c1e28a` |
| VideoPrism driver | `fd364264da0f668bbffe8959bdd353c323e1430ef2616aacc78580c206aeb01` |
| Ruri + EfficientAT driver | `479f566189af0dca7fbc9ce00a841221f6d745b10c3697b2bfbea0d7e11c4153` |
| Generated WAV fixture | `fd878fb5531d04a35fd3641f63bb6df1de628e49cc251ca465dc833a878ac8a9` |

Files remain outside Git at
`C:/CodexArtifacts/pretrained-reuse-r1/recheck-20261026-0015/`. The VideoPrism
recheck preserves its prior output digest
`df185755af2fabc6c8dcf0765743edd5cbb01cf6bb6bc7d968e469a929b9796f`.
MiniLM and MOMENT were absent at the expected cache paths and were not tested;
no runtime claim is added for them.

### Local checkpoint path recheck

The three weight files were also reopened directly from their retained local
artifact paths. Their byte counts and SHA-256 values match the table above:

| Component | Local checkpoint path | Bytes | SHA-256 |
|---|---|---:|---|
| VideoPrism-B | `C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz` | 457,475,296 | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` |
| Ruri v3-30m | `C:/CodexArtifacts/pretrained-reuse-r1/ruri-v3-30m/model.safetensors` | 146,828,152 | `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832` |
| EfficientAT DyMN-0.4 | `C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt` | 8,086,973 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` |

This is a read-only file-integrity recheck; it did not rerun model inference or
change any artifact.

## Decision

This closes only a fresh local runtime recheck for these three pinned frozen
components. It does not establish task quality, four-modality fusion, calibrated
caller-option probabilities, final package fit, or product adoption. The next
R2 quality run still requires an admitted task and split; the current
synchronized natural-AV source screens have not supplied one.

Primary source report on the independent branch:
[`R1_FROZEN_COMPONENT_CPU_RECHECK_20261026.md`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r0-r1-recheck-20261026/docs/pretrained_reuse/R1_FROZEN_COMPONENT_CPU_RECHECK_20261026.md).
