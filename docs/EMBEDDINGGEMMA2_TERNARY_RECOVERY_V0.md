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

## Phase 9 — portable bundle and paired evaluation

The selected ternary overlay and step-1,152 Recovery LoRA were exported to a
self-contained local bundle at
`C:/CodexArtifacts/embeddinggemma2-ternary-recovery-v0-attempt01-bundle`.
The bundle contains the pinned base checkpoint, processor/tokenizer, packed
ternary overlay, Recovery adapter, effective config, metadata, and license
notice. Its manifest SHA-256 is
`f7677723f4a642f4b666ac7feffa4a99209692eb9149e8a3611fc567b83c8546`; it records
1,706,973,718 total file bytes, including the 1,488,915,288-byte base, a
160,453,191-byte packed overlay plus scales, and a 20,232,264-byte adapter.
All bundle files were hash-verified. A fresh reload reproduced all 256 saved
best-validation option probabilities exactly (maximum absolute difference
0.0).

The unquantized diagnostic, ternary overlay, Recovery LoRA, and frozen Teacher
were paired on the same 2,917 examples, option order, and evaluation
preprocessing from the historical durable-Teacher evaluation. That evaluation
had already been observed; this is a reproducibility/comparison run, not a
blind audit and not a basis for tuning. The held-out records were not used for
model selection. The corpus SHA-256 is
`19812d5211a45e8eedee2aa0c6b36795f613d8a1f8e0592c881edfabd13a2d9d`.

| Candidate | Overall Acc. | Overall NLL | Overall Brier | Overall ECE | Macro modality Acc. | Macro NLL | Macro Brier | Macro ECE | Min modality Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher v0 | 0.7343 | 0.6380 | 0.3344 | 0.0344 | 0.6524 | 0.9000 | 0.4339 | 0.0463 | 0.4176 |
| Unquantized student | 0.4326 | 1.5164 | 0.7295 | 0.2466 | 0.4202 | 1.6717 | 0.7626 | 0.2407 | 0.3210 |
| Ternary student | 0.3157 | 1.6632 | 0.7479 | 0.0779 | 0.2387 | 1.8461 | 0.7973 | 0.0993 | 0.0625 |
| Ternary + Recovery LoRA | 0.3198 | 1.6032 | 0.7282 | 0.0572 | 0.2397 | 1.8068 | 0.7880 | 0.0945 | 0.0547 |

Per-modality results (Accuracy / NLL / Brier / ECE):

| Candidate | Audio | Image | Text | Video |
| --- | --- | --- | --- | --- |
| Teacher v0 | 0.9492 / 0.1854 / 0.0644 / 0.0295 | 0.5703 / 1.3744 / 0.5650 / 0.0529 | 0.6723 / 0.7323 / 0.4232 / 0.0442 | 0.4176 / 1.3080 / 0.6798 / 0.0589 |
| Unquantized student | 0.5654 / 2.1302 / 0.8611 / 0.4405 | 0.4297 / 2.1322 / 0.8602 / 0.2947 | 0.3645 / 1.0515 / 0.6300 / 0.1691 | 0.3210 / 1.3730 / 0.6990 / 0.0585 |
| Ternary student | 0.1191 / 2.4608 / 0.9276 / 0.0828 | 0.0625 / 2.3527 / 0.9128 / 0.1271 | 0.4890 / 1.0610 / 0.6034 / 0.0625 | 0.2841 / 1.5098 / 0.7452 / 0.1246 |
| Ternary + Recovery LoRA | 0.1055 / 2.3654 / 0.9122 / 0.0609 | 0.0547 / 2.3718 / 0.9154 / 0.1365 | 0.5060 / 1.0126 / 0.5718 / 0.0472 | 0.2926 / 1.4772 / 0.7525 / 0.1334 |

All student variants remain well below Teacher v0 and far below the four
90%-per-modality target. The Recovery adapter made small text/video and
calibration changes while further lowering Audio/Image accuracy versus the
ternary-only student. This is a mixed trade-off, not a successful four-modal
recovery or a reason to promote the artifact. The sealed audit remains unopened.

The full three-candidate student evaluation took 719.6 s unquantized, 709.2 s
ternary, and 780.6 s with Recovery on the RTX 3080 Laptop GPU. Peak allocated
VRAM was 2.56–2.58 GB per candidate; the evaluations used no cloud compute.
Video elapsed time dominated at about 502–512 s per candidate (352 examples,
about 1.43–1.45 s/example). These wall intervals include decoding,
preprocessing, inference, and output work; they are not GPU-kernel timings.

### Video preprocessing optimization

The evaluation CLI now has an opt-in, one-scene host-memory cache. It decodes
the full video with the pinned PyAV backend once, supplies the original video
metadata to the processor, and evicts the previous scene when the input path
changes. The default path and historical scores are unchanged. On a local
three-question CLEVRER scene, uncached processor input creation took 3.42 s
total; cached creation took 1.17 s total (about 66% lower). The `input_ids`,
`attention_mask`, `pixel_values_videos`, `video_position_ids`, and
`num_frames_per_video` tensors were bitwise equal for all three questions.
This CPU-side microbenchmark establishes input parity and repeated-decode
savings for one scene; a full paired evaluation has not yet been rerun with
the cache, so no full-run speedup is claimed. The cache holds at most one raw
scene to bound host/shared-memory use. A three-example Recovery inference smoke
then measured 5.52 s uncached and 2.83 s cached, with identical predictions
and maximum option-probability difference 0.0; peak allocated RTX VRAM was
2.58 GB. This is a small repeated-scene smoke, not a full-suite timing.

The machine reports approximately 16 GB of Intel Iris Xe shared system memory,
not 16 GB of dedicated integrated-GPU VRAM. The completed training and paired
evaluation used the RTX 3080 CUDA device. The active student environment did
not expose an Intel XPU device, so the integrated GPU was not used as a compute
device. No runtime upgrade or cloud run was used for this optimization check.

Phase 9 local outputs remain outside Git:

- Bundle: `C:/CodexArtifacts/embeddinggemma2-ternary-recovery-v0-attempt01-bundle`
- Reload validation: `C:/CodexArtifacts/embeddinggemma2-ternary-recovery-v0-attempt01-bundle-reload-validation`
- Paired evaluation: `C:/CodexArtifacts/embeddinggemma2-ternary-recovery-v0-attempt01-final-evaluation`
- Evaluation example hash: `19812d5211a45e8eedee2aa0c6b36795f613d8a1f8e0592c881edfabd13a2d9d`
- Prediction hashes (unquantized / ternary / recovery):
  `b62d6a849b964ecf1e4fcf6988569bdf9c3004c4482cb75d03a41497e10f5c3a` /
  `edcd32a502ac3b845c25152a6a60228f5c0cb13a0c82209ce2b7b57e03339b0a` /
  `508e08ac196fa38b54fbf4e0fe1dc21ce85b5e8dede9d76508c906edaf5d53c9`
