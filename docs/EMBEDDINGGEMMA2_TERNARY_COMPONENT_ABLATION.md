# EmbeddingGemma 2 ternary component ablation

## Scope

This is a no-training diagnostic of quantization damage in the selected
EmbeddingGemma 2 QAT checkpoint. It is not a sealed evaluation, a product
precision decision, or a new model artifact. Teacher v0, QAT attempt 08,
Recovery LoRA, their data, and their checkpoints were read-only. No Recovery
LoRA, sealed audit, or historical 2,917-example final evaluation was loaded.

## Integrity and execution

- Source branch/commit: `codex/qat-groupwise-performance` / `e66e4e92a38bfff3970818e9440ab70b763118c1`.
- Model: `google/embeddinggemma-2`, revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`; local BF16 base files were
  verified against the manifest.
- Selected QAT run: `embeddinggemma2-ternary-qat-v0-attempt08-workspace-warmstart`,
  best step 512. Metadata SHA-256:
  `2fe5f1d56d064098e1ec5da63d6c9095062017fd204ad1685ddc243fc1334e53`.
- BF16 selected shadow SHA-256:
  `9bc704d7c31e20cf66768e73a0bb82810aa98097a938202d46f1bcd8c5a84cae`.
- Selected packed overlay manifest SHA-256:
  `5a94951425b3913da5896a5a46be010e1f901f93d30291c4397407b2674d2edb`;
  tensor SHA-256:
  `7b90ea585e07653777d68c9f7022d705a74ddf4b8b404285bb8999fe82b74e34`.
- Fixed validation: 256 examples, 64 per modality. Snapshot SHA-256:
  `43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`;
  selection manifest SHA-256:
  `7aa1168ce3baa6402e0ed05b21038015c8dcc3607570e942f3a92ad667a9bc08`;
  ordered ID SHA-256:
  `a5af651b5d86dfc2ba0eddf0372ff148ec4e46dd31cf47eb0d5c24d4cfbb4319`.
- Command: `python scripts/diagnose_student_ternary_components.py` with the
  explicit paths and expected SHA-256 values above, `--temperature 0.1`,
  `--device cuda`, and a fresh output directory.
- Started `2026-10-09T01:30:42Z`; completed `2026-10-09T01:39:56Z`.
- Local runtime: RTX 3080 Laptop GPU, CUDA 12.4, PyTorch 2.6.0+cu124,
  Transformers 5.19.0, PEFT 0.21.2, PyAV 18.1.0. Video decoding used the
  installed Transformers fallback because torchcodec is absent. No cloud
  compute or training/optimizer update was used.
- Output directory:
  `C:/CodexArtifacts/embeddinggemma2-ternary-component-ablation-v0-attempt-06/`.
  It contains per-arm predictions and `diagnostic-results.json` with hashes,
  timings, GPU counters, full per-source metrics, and gradient audit details.
  Results JSON SHA-256:
  `759a0eaad8be78b74d05c15257a20ff28c012b56808deb8561a6ad378fb2b66c`.
  The companion `reproduction-metadata.json` records the full executed CLI
  arguments and input hashes; its SHA-256 is
  `5ae651111a4feedefa4e7ccf7badc44cbdf9212fafe46da9eb05f435d0a831b0`.
  The original results JSON predates serialization of the temperature field;
  the companion records the actual `0.1` argument without rewriting results.

The selected BF16 QAT shadow, rather than the original pretrained weights, is
the reference for every ablation. Each arm first applies the same packed
ternary overlay, then restores the chosen component's exact BF16 shadow values.
The reference and all four variants therefore share the same task-adapted QAT
checkpoint and validation IDs.

## Code and target audit

The loaded `EmbeddingGemma2Model` target inventory exactly matched the overlay:
483 tensors and 744,129,664 elements. Grouping includes each modality encoder
and its projection into the shared sequence:

| Component | Target tensors | Elements |
| --- | ---: | ---: |
| Audio path (`audio_tower.*`, `embed_audio.*`) | 149 | 305,497,216 |
| Vision path (`vision_tower.*`, `embed_vision.*`) | 115 | 167,706,624 |
| Shared decoder (`language_model.*`) | 219 | 270,925,824 |

The selector includes supported matrix weights and fails closed when it sees
an unclassified multidimensional parameter. One-dimensional parameters are
outside the selected ternary weight policy. Every selected shadow tensor was
re-quantized on CPU using the recorded group size and threshold; all 483 packed
codes and scales exactly matched the selected overlay. The observed trit counts
were 223,649,304 negative, 311,016,768 zero, and 209,463,592 positive.

The runtime forward path decodes the packed overlay into the loaded model's
selected parameter tensors; the modality processor path then feeds the
corresponding tower/projection and shared `language_model`. For a gradient-only
check, only the 483 ternary-target parameters were made trainable. One example
per modality produced finite, nonzero gradients in every target tensor on its
active modality path and the shared decoder (149/149 Audio, 115/115 Vision,
219/219 decoder). Inactive modality paths had no gradient. The target parameter
values were byte-for-byte unchanged, and no optimizer was constructed or
stepped. The isolated identity-STE unit test and diagnostic component-mapping
tests passed.

The CUDA allocator reported a 20.65 GB peak allocated and 21.95 GB peak
reserved against a 17.18 GB device total. Windows WDDM reports GPU memory
differently from dedicated VRAM, so these peak counters are not interpreted as
physical VRAM consumption; they are retained as raw diagnostics. The run
completed without OOM.
The four full validation arms took 507.36 seconds combined (roughly 125–129
seconds each); end-to-end diagnostic runtime was 9 minutes 14 seconds.

## Fixed-validation metrics

Each modality has 64 examples. ECE uses 15 bins. Values are computed from the
same 256 IDs, the same processor, and temperature 0.1.

| Variant | Modality | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| All ternary | Audio | 0.1406 | 2.3668 | 0.9155 | 0.0807 |
| All ternary | Image | 0.0781 | 2.3753 | 0.9154 | 0.1309 |
| All ternary | Text | 0.4062 | 1.0852 | 0.6548 | 0.1551 |
| All ternary | Video | 0.3281 | 1.3450 | 0.6912 | 0.1111 |
| Audio path BF16 | Audio | 0.0938 | 2.3781 | 0.9173 | 0.1256 |
| Audio path BF16 | Image | 0.0781 | 2.3753 | 0.9154 | 0.1309 |
| Audio path BF16 | Text | 0.4062 | 1.0852 | 0.6548 | 0.1551 |
| Audio path BF16 | Video | 0.3281 | 1.3450 | 0.6912 | 0.1111 |
| Vision path BF16 | Audio | 0.1406 | 2.3668 | 0.9155 | 0.0807 |
| Vision path BF16 | Image | 0.1406 | 2.4232 | 0.9219 | 0.0727 |
| Vision path BF16 | Text | 0.4062 | 1.0852 | 0.6548 | 0.1551 |
| Vision path BF16 | Video | 0.3125 | 1.3846 | 0.7143 | 0.1308 |
| Shared decoder BF16 | Audio | 0.1094 | 2.2861 | 0.8967 | 0.0043 |
| Shared decoder BF16 | Image | 0.1562 | 2.3157 | 0.9022 | 0.0394 |
| Shared decoder BF16 | Text | 0.4062 | 1.0552 | 0.6350 | 0.1895 |
| Shared decoder BF16 | Video | 0.2344 | 1.3805 | 0.7082 | 0.1096 |

| Variant | Macro accuracy | Macro NLL | Macro Brier | Macro ECE | Minimum modality accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| All ternary | 0.2383 | 1.7931 | 0.7942 | 0.1195 | 0.0781 |
| Audio path BF16 | 0.2266 | 1.7959 | 0.7947 | 0.1307 | 0.0781 |
| Vision path BF16 | 0.2500 | 1.8149 | 0.8016 | 0.1098 | 0.1406 |
| Shared decoder BF16 | 0.2266 | 1.7594 | 0.7855 | 0.0857 | 0.1094 |

## Interpretation and next step

Restoring only the Audio path to BF16 did not recover Audio; accuracy fell
from 14.1% to 9.4%. Restoring only Vision raised Image accuracy from 7.8% to
14.1%, but NLL and Brier worsened, so the additional correct predictions came
with less reliable probabilities. Restoring the shared decoder improved Image
accuracy to 15.6% and improved Audio NLL/Brier/ECE, but Audio accuracy fell to
10.9%, while Video accuracy fell from 32.8% to 23.4%. Text accuracy was
unchanged in all arms. No one-component BF16 restoration improved all four
modalities, and no arm clearly identifies the encoder towers as the sole
failure source.

The strongest next experiment is a **decoder-layer sensitivity ablation** on
the same fixed validation: restore one decoder block at a time to the selected
BF16 QAT shadow while leaving all other selected weights ternary. The shared
decoder restoration shifts several modalities at once, so identifying the
specific sensitive blocks is more actionable than changing the quantizer or
starting another QAT run now. Keep this diagnostic-only; do not promote a BF16
hybrid to the product model. Any later QAT change should be a separate short
experiment after the sensitive blocks are identified.

These are development-validation results selected after extensive prior use;
they are diagnostic only, not independent evidence of generalization. The
256-example snapshot is small, so differences of one or two examples should
not be overinterpreted. The 1.71 GB self-contained bundle issue remains
separate: the packed overlay still relies on loading the BF16 base at runtime.
