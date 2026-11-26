# R2 MiniWoB DINOv2 Frozen-Feature Path A Results (2026-11-12)

## Scope and disposition

This is a measured, CPU-only frozen-feature Decision experiment on the pinned
MiniWoB++ synthetic `click-color-v1:screen-only` task. It advances the screen
only R2 M-0/M-1 path from corpus preflight to an executed Path A readout. It is
not evidence for general image understanding, native video, audio, text, sensor
fusion, or the Omni product target. It does not close R2 M-2 or authorize
replacing the VideoPrism video path.

The task asks the model to select which of four screen-coordinate actions
targets the same color as a visible swatch. The model receives screenshot RGB
and normalized candidate coordinates. DOM, utterance, CSS names, audit labels,
and upstream target fields are excluded from model features. Opaque action IDs
are used only to map probabilities back into the output.

## Pinned inputs

- MiniWoB++ source commit: `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`, MIT.
- Frozen corpus: 512 train and 512 development-validation episodes, each with
  unique episode/screenshot identity. Train and validation have zero sample,
  group, image-byte, or decoded-pixel overlap.
- Train JSONL SHA-256:
  `be296443c53e29b77873856c710f48ef98ef21091e113a47108dbe04e60f54b9`.
- Validation JSONL SHA-256:
  `2be09a9f7f9a4d85f9d642f798b69d2fc39d5fcac0b0a0f9993d4fad892a350f`.
- Corpus manifest SHA-256:
  `dc9c7b5c3d7c7b044e319ef108da1a84ab24eedd7a10439942f8f5d05f17eecc`.
- Train sample-order SHA-256:
  `e1b4bc57576ec467ce30545eb03d8ce437b481d42df7095a3b8279218d6ea185`.
- Validation sample-order SHA-256:
  `e15b97f4129759e01949ddb5649da20d5e60854d751d865ad7cb1bc818e014f1`.
- Frozen encoder: `facebook/dinov2-small`, revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, Apache-2.0 per the pinned
  publisher card. Local weight SHA-256 is
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`;
  parameter count is 22,056,576. The local README, architecture config,
  processor config, and safetensors hashes were checked before loading. The
  checkpoint was loaded offline by the approved system Python with
  `trust_remote_code=False`; every parameter remained frozen.

The model processor's center crop would discard side content from these narrow
screens. The experiment therefore used a separately pinned, documented
whole-screen bicubic resize to 224x224, then the model-card RGB normalization.
This preprocessing choice is part of this candidate and is not claimed to be
equivalent to the upstream default processor. The DINO tokens were the CLS
token, mean patch token, and a 3x3 patch mean at each candidate center; the
candidate's normalized x/y were appended.

## Readout and controls

One shared option scorer was used for all four candidates:
`LayerNorm(1154) -> Linear(1154, 64) -> GELU -> Linear(64, 1)`. This gives
76,293 trainable parameters. The scorer is permutation-equivariant; the final
probabilities are serialized by opaque option ID.

Each learned arm used seed 17, AdamW, learning rate 0.001, weight decay 0.0001,
batch 64, up to 100 epochs, patience 15, `min_delta=0.0001`, and earliest
minimum development-validation NLL selection. No test or sealed-audit data was
loaded. Controls were frozen before scoring:

1. Train-derived candidate-position prior.
2. DINO features from a black screen, retaining candidate coordinates.
3. Train and validation screenshots independently deranged across episodes
   with a deterministic Sattolo cycle, retaining each row's candidates/label.
4. Same-example candidate-order permutation, mapping scores back by option ID.

## Results

The primary matched-screen arm selected epoch 99; the run continued through
epoch 100. Results use the same 512 development-validation IDs throughout.

| Arm / checkpoint | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Matched screen, selected epoch 99 | 0.9609 | 0.1231 | 0.0617 | 0.0506 | 0.9199 |
| Matched screen, final epoch 100 | 0.9570 | 0.1233 | 0.0627 | 0.0438 | 0.9207 |
| Position prior | 0.2422 | 1.3844 | 0.7491 | 0.0254 | 0.2676 |
| Black screen, selected epoch 1 | 0.2305 | 1.3863 | 0.7500 | 0.0209 | 0.2514 |
| Permuted screen, selected epoch 1 | 0.2441 | 1.3920 | 0.7528 | 0.0288 | 0.2729 |

The matched arm's selected training metrics were Accuracy 0.9961, NLL 0.0487,
Brier 0.0126 and ECE-15 0.0421. The development metrics are substantially
weaker, indicating a visible train/validation gap. The selected epoch was one
before the 100-epoch run end; this small endpoint difference does not establish
a broad learning plateau. The two controls that removed the correct screen
association stayed near four-choice chance, while matched screens scored
96.09%. This supports screen-content dependence on this synthetic task only.

After candidate-order permutation and ID remapping, the matched checkpoint had
the same accuracy, zero argmax changes, maximum probability difference
`1.385808e-6`, mean difference `8.063333e-9`, and 99th-percentile difference
`1.192093e-7`. A `2e-6` tolerance was applied to this post-fit CPU numerical
replay. The initial training process stopped after all arms had saved their
best/final checkpoints because its first assertion used a tighter `1e-7`
threshold. The finalized report was generated by replaying the saved feature
cache and checkpoint, not by retraining or changing the selected checkpoint.
Saved-best reload probabilities matched exactly (`0.0` max difference).

## Exposure, resources and artifacts

- Matched arm: 100 epochs / 51,200 row presentations total; selected epoch 99
  had consumed 50,688 rows. All 512 unique train screens were reused.
- Black-screen arm: 16 epochs / 8,192 presentations; selected epoch 1 had
  consumed 512. The arm stopped after its fixed patience interval.
- Permuted-screen arm: 16 epochs / 8,192 presentations; selected epoch 1 had
  consumed 512. The arm stopped after its fixed patience interval.
- Total across all three arms: 67,584 training-row presentations, with 512
  unique source screens per arm. Validation count was 512, never used for
  gradient updates.
- Feature extraction: 93.18 seconds. Readout fitting/evaluation across arms:
  11.25 seconds of recorded epoch time. Python 3.11.9, PyTorch 2.6.0+cu124,
  Transformers 5.6.2, NumPy 2.4.6, Pillow 10.4.0; CPU, two threads. CUDA and
  cloud compute were not used. The process RSS was observed at about 0.73 GiB
  during the run; peak RAM was not instrumented.
- The RTX 3080 was occupied by a separate Orbitune run; its process was left
  untouched. This experiment did not initialize CUDA or consume VRAM.
- A single matched image-path package would contain 22,056,576 frozen encoder
  parameters plus 76,293 readout parameters (22,132,869 total); the DINO
  weights (88,249,960 bytes) plus selected readout (305,644 bytes) total
  88,555,604 bytes. This is only the image path, not a four-modality model.

External run directory:
`D:\CodexArtifacts\pretrained-reuse-r2\miniwob-dino-path-a-v1-attempt-01\`.
It contains effective config, source/corpus/model hashes, cached frozen
features, all three arms' best and final weights, per-epoch histories,
per-example probabilities, and finalized controls. Key artifact hashes:

- Executed effective config: `97fc8827ec0f322800560d9853fd10304d90bb414ae2ae14cc0e942b23a737a3`.
- Config file hash recorded by the run: `374beac974125816bbaeb4a977f1ae1c14cec9533f7cf4f50a98673dcc2c594e`;
  corrected checked-in config SHA-256: `84647904421e81373bcccca4b15a40bd8878325c67f24a4351d2f61ba2fec6d7`.
- Frozen feature cache: `051ff80289b2b5a3232db524486bed8f6b939938a24358707758e4d62fe7eaef`.
- Final experiment results: `f3ee110baf0fe8b9f5648f2db2418f14c4de089d00db4effdd42b10123f71685`.
- Matched selected readout: `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`.
- Matched final readout: `bc2ed4ea4aa2dc35a5cadae5e7a91da4661a601cf1f2c61fb8b933235e2834f4`.

The run began at `2026-11-12 20:56:34 +0900`; the training process's exact
terminal timestamp was not recorded. Post-fit artifact reconciliation finished
at `2026-11-12 21:02:05 +0900`.

The code commit recorded at fit time was `f328eb1c2b09ab49406510e5c9514970137d684b`.
Its source-file-set hash is `96fadd1609e903bf0566312509df4a2c260ad1bb6e0630b3e438452f8b79e43a`.
The config text was corrected after execution to describe the Sattolo derangement
accurately and to document the post-fit numerical tolerance. The immutable
executed effective config and its SHA above retain what the run actually used.

## Interpretation and next gate

This is a successful Path A implementation/evaluation for one synthetic
screen-only task, not proof of 95–98% quality for real images or other
modalities. The train/validation gap, generated color-matching task, single
seed, development-only validation, and custom whole-screen resize limit the
claim. No product candidate was selected and no existing VideoPrism, v1/v2,
sealed-audit, or legacy artifact was changed.

Continue the Roadmap's VideoPrism runtime-admission and rights-cleared joint
audio/video/sensor task gates. Keep this DINO experiment as a separate image
candidate and compare it only under a new predeclared task/architecture
protocol; do not use its result to bypass the pinned VideoPrism runtime gate or
to close the both-input-required M-2 gate.
