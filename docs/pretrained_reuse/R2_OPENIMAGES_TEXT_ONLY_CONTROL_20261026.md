# Open Images frozen-feature text-only control — 2026-10-26

## Result

A separately fitted readout that receives **only** the question and candidate
text reached 75.60% Accuracy on the same 168-example development validation
set. It was trained once for a fixed eight epochs and evaluated at epoch 8;
there was no checkpoint selection or calibration. Compared with the selected
image+text readout, its predictions agreed on 161/168 rows. All six net
correctness differences favored the image+text-trained scorer (133 versus 127
correct); the text-only scorer corrected none of its errors. This comparison
is descriptive only: the image+text checkpoint was selected by validation NLL,
while the text-only control used a fixed endpoint, and the readout capacities
also differ.

| Model/input | Resident parameters | n | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected frozen VideoPrism + MiniLM readout; aligned image | 131,902,721 | 168 | 79.17% | 0.5609 | 0.3014 | 0.1454 | 0.6931 |
| MiniLM-only frozen text features + fixed-final readout | 17,439,233 | 168 | 75.60% | 0.6419 | 0.3366 | 0.1912 | 0.6237 |

The original image+text run selected epoch 8, and its best and final state
tensors were verified exactly equal. The first row is still validation-selected,
while the text-only control uses a fixed endpoint. The second row is one
task-shortcut diagnostic, not a new model candidate. It uses 49,409 readout
parameters versus the original 147,713, so the comparison is not
parameter-matched. The original selected readout's own zero-image and
shuffled-image interventions preserved every
top-1 prediction (see the [input-control report](R2_OPENIMAGES_INPUT_CONTROLS_20261026.md)).
Taken together, the scores show that the task provides strong question/candidate
text signal, and that the original scorer did not need image features at
inference to preserve any of its choices. They do **not** quantify a causal
training benefit from image features.

## Fixed eight-epoch curve

| Epoch | Train Accuracy | Train NLL | Validation Accuracy | Validation NLL | Validation Brier | Validation ECE-15 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 63.10% | 1.3299 | 73.21% | 1.3134 | 0.7126 | 0.4606 |
| 2 | 66.67% | 1.2639 | 72.62% | 1.2265 | 0.6657 | 0.4269 |
| 3 | 66.67% | 1.1891 | 72.62% | 1.1285 | 0.6102 | 0.3905 |
| 4 | 66.67% | 1.1057 | 72.62% | 1.0203 | 0.5463 | 0.3467 |
| 5 | 69.05% | 1.0180 | 75.00% | 0.9074 | 0.4789 | 0.3097 |
| 6 | 69.64% | 0.9337 | 75.60% | 0.8008 | 0.4174 | 0.2509 |
| 7 | 71.43% | 0.8604 | 75.60% | 0.7111 | 0.3697 | 0.1937 |
| 8 | 72.02% | 0.8015 | 75.60% | 0.6419 | 0.3366 | 0.1912 |

This short curve shows no obvious training/validation inversion, but it is
insufficient to infer generalization. Validation metrics were recorded during
the run but did not select or alter the fixed endpoint.

## Fixed diagnostic protocol

- Exactly 168 existing training rows and 168 existing validation rows were
  used, with the original sample order and source/image grouping. No new media
  or labels were read. The validation split was already exposed and used for
  selecting the earlier image+text checkpoint.
- The text encoder was the pinned frozen MiniLM-L3 revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; its weights SHA-256 is
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` and its
  parameter count is 17,389,824.
- The trainable readout receives 384-D candidate-text features only; no image
  feature or image checkpoint enters training or inference. It has 49,409
  parameters; frozen text plus readout totals 17,439,233.
- Seed 17, AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, and
  the same eight-epoch shuffled-example schedule as the image+text run were
  fixed in advance. The single final epoch was scored; no validation selector,
  seed search, or temperature fit was used.
- Save/reload validation-logit difference was exactly 0.0. Training took
  0.065 seconds after 0.149 seconds of text feature extraction on CPU. No GPU,
  cloud compute, sealed audit, or existing checkpoint modification was used.
- This remains a task-construction diagnostic on exposed validation rows, not
  independent text quality, image quality, or product evidence.

## Provenance

- Source commit: `b573ffce54d50677ffa7f8186625a686c8de3d44`.
- Runner SHA-256:
  `e507d391eee76689f31dba8fedc391c806ff465968bccfcdf6835e3d025142a1`.
- Task config SHA-256:
  `3cca69b55316b2efeeb1d458025cf79a53dbc7a0e44d4c94183dba53d4ae225b`.
- Training rows SHA-256:
  `2cc95cfa92369938611a4519801b332b6d3c623bdb1414cb9f16df0e7e003309`.
- Validation rows SHA-256:
  `c2396b3974de91afea66424b201388fb6d47013098a211d31b0191bd0a9adfb7`.
- Validation feature source SHA-256:
  `2b13b7b50e026aa2c8e975239784b1d4aac5ea6343a446ba0bf490ca15fe1622`.
- Validation sample-order SHA-256:
  `785c3cd1b6fb260964b615fd27843063671a379baead722ae6a7b375ef8e1636`.
- External result directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\openimages-videoprism-text-only-control-20261026-attempt-01\`.
- `run-metadata.json` SHA-256:
  `cf770a0fc4136ebbd92e9437b6f6dd4bbe16b72d21c6c7f593351222c09cfe45`.
- Final readout SHA-256:
  `f67cc6469ed7ec073be2d2c7f47dd09926097e4472bdddd3009b1df33ca5c80c`.
- Validation predictions SHA-256:
  `f62e996072bde6a943f5cca4bd8d8e83772da0832b48b1cff013d87ecf8a271f`.

The control does not rehabilitate the original 79.17% as image-grounded
quality. The next image experiment requires a new task and validation split
whose target cannot be recovered from candidate-text regularities; it must
predeclare image-zero and image-shuffle controls and must not reuse these
exposed rows for model selection.
