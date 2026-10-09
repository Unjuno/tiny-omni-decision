# EmbeddingGemma 2 Decision Head Readout Diagnostic

## Status

Completed as a frozen-backbone, validation-only diagnostic. Neither head is adopted. No QAT, Recovery LoRA, Decision Head product promotion, or sealed-audit evaluation was performed.

## Reproduction

- Branch: `codex/decision-head-validation`
- Source commit for completed run: `00d16a5ee708a277f59acf71264533db4c8806a7`
- Run: `embeddinggemma2-decision-head-v0-attempt-03`
- Started: 2026-10-09 04:42:33 UTC; ended: 2026-10-09 05:33:54 UTC (51m 21s)
- Command: `python scripts/train_student_decision_head.py --config configs/student/embeddinggemma2_decision_head_v0.yaml`
- Effective config SHA-256: `c1829cb0184987395f75de2427de2a4a01650c8d3721a1a769e4b638d63f718c`
- Results SHA-256: `0160a5af6642162613ab35198bdc719c63fa64d49e520d8fdd938ce425415615`
- Output directory: `C:\CodexArtifacts\embeddinggemma2-decision-head-v0-attempt-03`
- Model: `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`; 744,371,488 backbone parameters, BF16.
- Ternary overlay: QAT attempt 08 selected step 512; overlay manifest SHA-256 `5a94951425b3913da5896a5a46be010e1f901f93d30291c4397407b2674d2edb`, tensor SHA-256 `7b90ea585e07653777d68c9f7022d705a74ddf4b8b404285bb8999fe82b74e34`.
- Training corpus: 4,859 examples; corpus SHA-256 `c3e6796a2df47471d42e671272b3050f9db150ac6b7ef7289e3902253f36d84d`; frozen order SHA-256 `d12e2225776fe895efd9c9686742326e494170ee4291506649f75dd70b7afbac`. All 4,859 IDs were unique and each was consumed once per backbone.
- Fixed validation: 256 examples, 64 per modality; snapshot SHA-256 `43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`; ordered ID SHA-256 `a5af651b5d86dfc2ba0eddf0372ff148ec4e46dd31cf47eb0d5c24d4cfbb4319`.
- Teacher option cache SHA-256: `f2d3cc1a9d579c7fe75f0dfb1e5441b81964a6d41e8efa76ed1cc43bc0d3e0a4`.
- Environment: Windows 10 build 26300; Python 3.11.9; PyTorch 2.6.0+cu124; CUDA runtime 12.4; Transformers 5.19.0; PEFT 0.21.2; PyAV 18.1.0; NVIDIA RTX 3080 Laptop 16 GB. Initial free GPU memory 16,008,609,792 bytes. Video decode used the existing torchvision fallback. No environment packages were upgraded.
- Sealed audit: not accessed. The same fixed validation IDs and evaluation preprocessing were used for each baseline/head pair.

## Method

For each frozen backbone separately, the runner extracted one query embedding per example and cached option embeddings in first-occurrence order within the split. Video validation used the same per-scene frame cache semantics as the established diagnostic. The completed ternary cosine predictions exactly reproduced the historical all-ternary diagnostic: 0/256 top-1 disagreements and maximum absolute probability difference 0. This is the parity gate for interpreting attempt 03.

The 768D query and option vectors feed `[q, o, q*o] → MLP(128, GELU) → scalar score`. The head has 295,169 trainable FP32 parameters (1,181,180-byte safetensors artifact, including metadata). Each variant received one ordered pass over the same 4,859 examples: AdamW, LR `3e-4`, weight decay `0.01`, and loss `option KL + 0.2 CE + 0.2 Brier`. Only the head was trainable; backbone trainable parameters were zero and no backbone gradients were present. Each head artifact was reloaded and verified.

An initial attempt 01 used batched option/text feature extraction. Its ternary cosine predictions disagreed with the established diagnostic on 184/256 rows (maximum probability difference 0.492); its metrics are invalid for comparison and are excluded below. The cause was batch-context sensitivity of ternary embeddings. Attempt 02 then stopped at 4,300/4,859 train examples because one sample had only one uncached option and the shared helper requires at least two. Its incomplete output is retained separately and not used. A regression test and single-option processor fallback were committed before attempt 03. Attempt 03 completed successfully.

## Validation results

Each cell is `cosine → trained head`. Accuracy is a fraction; NLL, Brier, and ECE use the runner's fixed definitions. There are 64 examples in every modality.

### Ternary backbone

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 0.141 → 0.172 | 2.367 → 2.289 | 0.916 → 0.897 | 0.081 → 0.056 |
| Image | 0.078 → 0.109 | 2.375 → 2.314 | 0.915 → 0.903 | 0.131 → 0.063 |
| Text | 0.406 → 0.594 | 1.085 → 0.974 | 0.655 → 0.551 | 0.155 → 0.179 |
| Video | 0.328 → 0.297 | 1.345 → 1.355 | 0.691 → 0.696 | 0.111 → 0.130 |
| Macro (4 modalities) | 0.238 → 0.293 | 1.793 → 1.733 | 0.794 → 0.762 | 0.119 → 0.107 |

The fixed adoption rule required both macro NLL and macro Brier to improve and no modality accuracy to fall by more than one example out of 64. NLL and Brier improved, but Video lost 2/64 examples; therefore the ternary head is **not adopted**. Head-only scoring latency was p50 0.308 ms and p95 0.438 ms across 256 choice sets; cosine scoring was 0.135 s total and head scoring 0.144 s total in that pass. This measures readout scoring on cached vectors, not end-to-end multimodal latency.

### Unquantized backbone

| Modality | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Audio | 0.594 → 0.094 | 2.128 → 2.287 | 0.861 → 0.897 | 0.468 → 0.110 |
| Image | 0.469 → 0.063 | 2.118 → 2.330 | 0.857 → 0.906 | 0.333 → 0.176 |
| Text | 0.406 → 0.563 | 1.055 → 0.959 | 0.635 → 0.547 | 0.194 → 0.102 |
| Video | 0.313 → 0.391 | 1.323 → 1.341 | 0.682 → 0.697 | 0.081 → 0.101 |
| Macro (4 modalities) | 0.445 → 0.277 | 1.656 → 1.729 | 0.759 → 0.762 | 0.269 → 0.122 |

The unquantized head worsened macro accuracy by 16.8 percentage points and macro NLL/Brier; it is **not adopted**. Lower ECE alone does not compensate for the loss in accuracy and proper scoring rules. Head-only scoring latency was p50 0.313 ms and p95 0.396 ms; cosine scoring took 0.205 s total and head scoring 0.114 s total for the 256 cached choice sets.

## Size and interpretation

The head adds 295,169 parameters and 1,181,180 bytes per backbone. The pinned base directory is 1,525,788,471 bytes. The packed ternary overlay directory is 160,920,316 bytes but is not a standalone model; base plus overlay plus head is 1,687,889,967 logical payload bytes. The unquantized base plus head is 1,526,969,651 bytes. No runtime memory or end-to-end speedup claim is made.

The ternary result is a validation-only trade-off: text, image, and audio improve in this sample while video weakens slightly, and the fixed safety rule rejects the candidate. The unquantized result shows substantial forgetting on Audio and Image despite Text/Video Accuracy gains. These results do not establish a general limit of the architecture and do not justify quantization changes, another training stage, or product promotion. No additional seed was run.

## Artifacts and integrity

- `results.json`: SHA-256 `0160a5af6642162613ab35198bdc719c63fa64d49e520d8fdd938ce425415615`.
- `effective-config.yaml`: SHA-256 `c1829cb0184987395f75de2427de2a4a01650c8d3721a1a769e4b638d63f718c`.
- Ternary head: SHA-256 `b78650817ffb3dd9943c3c9cef1947051220f94efad8e20dd434cb1b1a726698`.
- Unquantized head: SHA-256 `1f4754c2e68e64951a0e65c63265967fee20de4e95cef0656f39cc17ca969542`.
- Frozen feature caches, per-example predictions, and full one-pass loss histories remain under the attempt-03 output directory. They are external local artifacts and were not added to Git.
- Attempts 01 and 02 were not overwritten. The QAT checkpoint, overlay, model files, corpora, and Teacher artifacts were read-only throughout.
