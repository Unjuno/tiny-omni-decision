# EmbeddingGemma 2 ternary decoder-depth sensitivity

## Result and scope

This no-training experiment divided the 24 shared decoder blocks into three
equal contiguous groups. For each arm, exactly one group's BF16 values were
restored from the selected QAT step-512 shadow; all other ternary targets,
including decoder weights outside those blocks, remained ternary. All 256
frozen development-validation IDs were evaluated for every arm. No QAT,
Recovery LoRA, Decision Head, sealed audit, or historical 2,917-example final
evaluation was used.

No group met the preregistered clear-improvement condition: macro NLL and macro
Brier both had to improve by at least 1% relative to all-ternary, and no
modality could lose more than one correct example out of 64. **Stop layer-level
search here.** Late blocks showed a small mixed signal, but the NLL change was
below threshold and Audio lost two examples. The results do not justify
promoting a BF16 hybrid or beginning a new training run.

## Reproducibility

- Branch/commit: `codex/qat-groupwise-performance` /
  `bfe4fb77b3e0d0c5cebd469746a2167775e3a28c`.
- Runner: `scripts/diagnose_student_decoder_depth_ablation.py`;
  SHA-256 `e14ca39c0b27b5041d46426c7ffacf1c8588f3d54dc325899f77e97891d392d0`.
- Base: `google/embeddinggemma-2`, revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- QAT run: attempt 08, selected step 512; metadata SHA-256
  `2fe5f1d56d064098e1ec5da63d6c9095062017fd204ad1685ddc243fc1334e53`.
- Selected BF16 shadow SHA-256:
  `9bc704d7c31e20cf66768e73a0bb82810aa98097a938202d46f1bcd8c5a84cae`.
- Packed overlay manifest SHA-256:
  `5a94951425b3913da5896a5a46be010e1f901f93d30291c4397407b2674d2edb`;
  overlay tensor SHA-256:
  `7b90ea585e07653777d68c9f7022d705a74ddf4b8b404285bb8999fe82b74e34`.
- All-ternary component-diagnostic baseline SHA-256:
  `759a0eaad8be78b74d05c15257a20ff28c012b56808deb8561a6ad378fb2b66c`.
- Validation snapshot SHA-256:
  `43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`;
  selection manifest SHA-256:
  `7aa1168ce3baa6402e0ed05b21038015c8dcc3607570e942f3a92ad667a9bc08`;
  ordered ID SHA-256:
  `a5af651b5d86dfc2ba0eddf0372ff148ec4e46dd31cf47eb0d5c24d4cfbb4319`.
- Score temperature 0.1; ECE uses 15 bins. All modalities have 64 examples.
- Local run: RTX 3080 Laptop, CUDA 12.4, PyTorch 2.6.0+cu124,
  Transformers 5.19.0, PEFT 0.21.2, PyAV 18.1.0. The video processor used
  the installed torchvision fallback because torchcodec is absent.
- Started `2026-10-09T02:20:57Z`, completed `2026-10-09T02:27:18Z`.
  Validation arms took about 121–125 seconds each. No cloud compute was used.
- Output directory:
  `C:/CodexArtifacts/embeddinggemma2-ternary-decoder-depth-ablation-v0-attempt-01/`.
  Machine-readable result SHA-256:
  `387738de6022f2bc61ea856320b8debd3db31799791a494e34dd3b09d0ccbe3f5`.
  Exact input paths, hashes, and CLI arguments are in the sibling
  `reproduction-metadata.json` (SHA-256
  `e1b2147c640cbbca6d8857fed5d5621e70d7cb71e13bfb1f4243fbc0cd72be9c`).

The invocation was `python scripts/diagnose_student_decoder_depth_ablation.py`
with the explicit paths and expected hashes recorded in that reproduction
metadata file. The runner verifies the baseline, shadow, overlay, base model,
validation corpus, ordered IDs, and local media before evaluating. Prediction
files are saved separately for each arm.

## Group construction

Groups were derived from actual selected target names, not guessed module
paths. The loaded model has `language_model.layers.0` through `.23`; each block
has nine selected target tensors. Decoder targets outside these blocks were
left ternary in every arm.

| Group | Layer IDs | Restored tensors | BF16 elements |
| --- | --- | ---: | ---: |
| Early | 0–7 | 72 | 42,991,616 |
| Middle | 8–15 | 72 | 42,991,616 |
| Late | 16–23 | 72 | 44,040,192 |

The packed overlay was re-quantized against the selected BF16 shadow and all
483 target tensors matched exactly before these runs. Each restore was checked
bitwise against that same shadow. The script performed zero optimizer updates.

## Metrics

Macro scores are the unweighted mean of the four modality scores. Accuracy,
NLL, Brier, and 15-bin ECE are listed per modality.

| Variant | Modality | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| All ternary baseline | Audio | 0.1406 | 2.3668 | 0.9155 | 0.0807 |
| All ternary baseline | Image | 0.0781 | 2.3753 | 0.9154 | 0.1309 |
| All ternary baseline | Text | 0.4062 | 1.0852 | 0.6548 | 0.1551 |
| All ternary baseline | Video | 0.3281 | 1.3450 | 0.6912 | 0.1111 |
| Early BF16 | Audio | 0.1250 | 2.3353 | 0.9053 | 0.0369 |
| Early BF16 | Image | 0.1562 | 2.3568 | 0.9058 | 0.0168 |
| Early BF16 | Text | 0.2500 | 1.1734 | 0.7168 | 0.2742 |
| Early BF16 | Video | 0.3125 | 1.3976 | 0.7210 | 0.0961 |
| Middle BF16 | Audio | 0.1250 | 2.3379 | 0.9035 | 0.0579 |
| Middle BF16 | Image | 0.1562 | 2.4672 | 0.9227 | 0.0258 |
| Middle BF16 | Text | 0.4219 | 1.0251 | 0.5898 | 0.1273 |
| Middle BF16 | Video | 0.2812 | 1.4105 | 0.7212 | 0.1358 |
| Late BF16 | Audio | 0.1094 | 2.3473 | 0.9006 | 0.1010 |
| Late BF16 | Image | 0.1250 | 2.3651 | 0.9109 | 0.0577 |
| Late BF16 | Text | 0.4688 | 1.0847 | 0.6260 | 0.1392 |
| Late BF16 | Video | 0.3438 | 1.3261 | 0.6958 | 0.1649 |

| Variant | Macro Accuracy | Macro NLL | Δ NLL vs baseline | Macro Brier | Δ Brier vs baseline | Worst Accuracy change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All ternary baseline | 0.2383 | 1.7931 | — | 0.7942 | — | — |
| Early BF16 | 0.2109 | 1.8157 | +1.26% | 0.8122 | +2.26% | −15.62 pp (Text) |
| Middle BF16 | 0.2461 | 1.8102 | +0.95% | 0.7843 | −1.25% | −4.69 pp (Video) |
| Late BF16 | 0.2617 | 1.7808 | −0.68% | 0.7834 | −1.37% | −3.12 pp (Audio) |

Positive Δ indicates a worse loss; negative Δ indicates a lower loss. The late
group's NLL improvement is 0.0123 absolute and its Brier improvement is 0.0109
absolute. It also changes only a few examples in several modalities, which is
not strong evidence on this repeatedly inspected validation set.

## Decision

Early restoration is worse on both macro loss measures and macro Accuracy.
Middle restoration improves macro Brier but worsens macro NLL and lowers Video
Accuracy by three examples. Late restoration improves macro Accuracy and both
losses directionally, but its NLL gain is below the 1% threshold and Audio
Accuracy falls by two examples. Therefore no arm satisfies the fixed
clear-improvement rule. Stop layer-level BF16 restoration search here, as
planned.

If work continues later, change one axis: evaluate a quantization-method or
readout hypothesis in a separate experiment. Do not choose a product precision
mix or launch QAT/Recovery training based on these small validation shifts. The
separate 1.71 GB runtime bundle still depends on the BF16 base.
