# EmbeddingGemma 2 ternary Recovery LoRA v0

## Scope and provenance

This report records one local Recovery LoRA run on the frozen selected ternary
overlay from QAT attempt 08. It is an experiment, not a product Teacher and not
a deployment-ready student. No sealed audit data was loaded. The Teacher v0,
QAT attempt 08, their corpora, and their checkpoints remain read-only.

- Source commit: `92369a487eb602fe03a4584dafad1b0d99aa6cf0`
- Experiment: `embeddinggemma2-ternary-recovery-v0-attempt01-lora16-decoder-all`
- Started: `2026-10-08T22:11:51.125234+00:00`
- Completed: `2026-10-08T23:40:05.754901+00:00`
- Output directory: `C:/CodexArtifacts/embeddinggemma2-ternary-recovery-v0-attempt01-lora16-decoder-all`
- Config SHA-256: `01456db7acc21ce331d8fd034e7814952c1ca6329bbc3c11259eb05beb37d4f8`
- Training script SHA-256: `9938aff609e112d402e689f15507215cd284ac012b74944564742b8c65a44519`
- Recovery module SHA-256: `bd16c4c9357b1aab92e5b0f6e38cd013f8436e3b85a8d8950ce3f6dbaedfbfcf`
- Train corpus SHA-256: `c3e6796a2df47471d42e671272b3050f9db150ac6b7ef7289e3902253f36d84`
- Validation snapshot SHA-256: `43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`
- Ordered validation ID SHA-256: `a5af651b5d86dfc2ba0eddf0372ff148ec4e46dd31cf47eb0d5c24d4cfbb4319`
- Teacher option cache SHA-256: `f2d3cc1a9d579c7fe75f0dfb1e5441b81964a6d41e8efa76ed1cc43bc0d3e0a4`
- Source QAT run metadata SHA-256: `2fe5f1d56d064098e1ec5da63d6c9095062017fd204ad1685ddc243fc1334e53`
- Source selected QAT best: step 512; shadow SHA-256
  `9bc704d7c31e20cf66768e73a0bb82810aa98097a938202d46f1bcd8c5a84cae`
- Frozen ternary overlay manifest SHA-256:
  `5a94951425b3913da5896a5a46be010e1f901f93d30291c4397407b2674d2edb`
- Frozen ternary overlay tensor SHA-256:
  `7b90ea585e07653777d68c9f7022d705a74ddf4b8b404285bb8999fe82b74e34`

## Method

The pinned EmbeddingGemma 2 BF16 base was loaded, the selected QAT ternary
overlay applied, and a fresh rank-16 LoRA was attached only to actual
`language_model.layers.*` decoder attention and MLP projections. Vision, audio,
and other modules were excluded. The ternary base was frozen; 5,046,272 LoRA
parameters were trainable (about 0.68% of the 744,371,488-parameter base).
The adapter uses alpha 32 and dropout 0.05. Training used seed 17, AdamW at
`5e-5`, cosine schedule, 3% warmup, gradient accumulation 4, and the existing
option KL + CE + Brier objective. No Teacher model was loaded; the frozen
Teacher's option cache provided training targets.

The corpus contained 4,859 examples: Audio 1,024; Image 256; Text 2,875
(Open-Jev 1,977 and Typed Decisions Synth 898); Video 704. One pass consumed
4,859 unique examples in 4,859 microbatches and 1,215 optimizer updates. There
was no sample repeat. Validation used the fixed 256-example snapshot (64 per
modality) and the fixed ID/order hash above. Checkpoints and evaluations were
written every 128 updates, with a final evaluation at step 1,215. The selector
minimized macro-modality NLL, then maximized minimum modality accuracy, then
minimized macro Brier.

## Results

The best checkpoint was step 1,152. It was reloaded and re-evaluated; the
maximum probability difference from the saved best evaluation was recorded as
`best_reload_max_probability_difference` in `run-metadata.json`. The final
step-1,215 checkpoint was retained separately and did not overwrite the best.

| Validation checkpoint | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Audio Acc. | Image Acc. | Text Acc. | Video Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial ternary, step 0 | 0.2383 | 1.7931 | 0.7942 | 0.1195 | 0.1406 | 0.0781 | 0.4063 | 0.3281 |
| Best, step 1,152 | 0.2578 | 1.7688 | 0.7863 | 0.1243 | 0.1250 | 0.0469 | 0.5469 | 0.3125 |
| Final, step 1,215 | 0.2422 | 1.8087 | 0.7909 | 0.1190 | 0.0781 | 0.0938 | 0.4219 | 0.3750 |

Best-step per-modality metrics (Accuracy / NLL / Brier / ECE; 64 examples each):

| Modality | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Audio | 0.1250 | 2.3211 | 0.9005 | 0.0583 |
| Image | 0.0469 | 2.3171 | 0.9125 | 0.1544 |
| Text | 0.5469 | 1.0146 | 0.5800 | 0.1086 |
| Video | 0.3125 | 1.4224 | 0.7523 | 0.1760 |

On the same validation examples, frozen Teacher v0 scored 0.6602 overall
accuracy (Audio 0.9219, Image 0.5938, Text 0.7188, Video 0.4063). Recovery
improved Text relative to the initial ternary model, while Audio and Image
remained very weak and aggregate results stayed far below the Teacher. This
is not evidence that more steps, a different adapter, or another method cannot
work; it shows that this single configuration and one corpus pass did not
recover useful four-modality quality. The 64-example modality slices are
small, and repeated validation informed checkpoint selection.

The complete learning curve, per-source/per-modality metrics, predictions,
sampling counts, effective config, adapter hashes, and environment are in the
external run directory. Rolling training loss is not a fixed-example
generalization gap and should not be compared as one.

## Runtime and environment

- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, reported total 17,179,344,896 bytes
- Peak allocated VRAM: 4,213,123,584 bytes; peak reserved: 5,536,481,280 bytes
- Wall time: 5,288.3 s (about 88.1 min); validation evaluation time total: 1,348.6 s
- Local compute cost: $0 cloud spend
- Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.19.0,
  PEFT 0.21.2
- The isolated student venv needed `peft==0.21.2`; it was installed without
  dependency upgrades. No other runtime package was upgraded for the run.

The machine also has an Intel Iris Xe integrated GPU. Its shared system-memory
budget is not dedicated CUDA VRAM and this training implementation used the
NVIDIA CUDA device. The run's VRAM measurements therefore describe the RTX
3080, not a combined or pooled GPU memory space.

## Integrity and status

Run metadata records state `complete`, 1,215 planned and completed updates,
4,859 consumed and unique examples, no loaded sealed audit, and
`product_teacher_promotion: false`. The best and final adapter files are each
20,232,264 bytes; they remain outside Git. No Teacher or QAT artifact was
overwritten. This result is a validation-only research artifact and must not
be treated as a production promotion or a sealed-audit claim.
