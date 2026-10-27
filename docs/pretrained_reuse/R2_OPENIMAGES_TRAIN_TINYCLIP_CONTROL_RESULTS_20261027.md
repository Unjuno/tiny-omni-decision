# TinyCLIP on four-choice train-candidate image controls

The [fixed protocol](R2_OPENIMAGES_TRAIN_TINYCLIP_CONTROL_PROTOCOL_20261027.md)
ran the previously audited 83,101,185-parameter TinyCLIP revision on CPU,
without training, using the same 257 train-candidate four-choice tasks used by
the option-only diagnostic. Media hashes and matching page declarations were
checked before inference. The pretrained model's possible YFCC15M source
overlap is unknown. This is development data, not independent validation.

| Input condition | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Correct image | 0.5370 | 1.3077 | 0.6513 | 0.1736 |
| Fixed deranged image | 0.1984 | 2.2981 | 1.0754 | 0.4375 |
| Uniform image at processor mean | 0.1946 | 1.8965 | 0.9641 | 0.3528 |
| Cross-fold option-only prior | 0.4397 | 1.2469 | 0.6733 | 0.0887 |

The correct-image model changed correctness on 127/257 examples when images
were deranged and 142/257 when images were replaced by the uniform image.
Top-1 agreement with aligned images was 28.8% and 24.1% respectively. Thus the
fixed model responds substantially to the image intervention on this task.
The option-only method is a separately fitted, cross-fold diagnostic, not a
pretrained model: it had lower NLL despite lower Accuracy. Neither is a
calibrated product decision system.

Aligned Accuracy varied strongly by target: Plant 5/26, Clothing 8/27,
Vehicle 9/25, Sculpture 14/15 and Door 17/23. These small groups and the
option-role shortcut limit interpretation. The image controls do not resolve
the label prior, copyright ownership, source overlap, task ambiguity, unseen
sources, or multimodal fusion. No benchmark was filtered and no checkpoint,
prompt, temperature or model architecture was selected using these outcomes.
Image scoring took 14.30 seconds and the uniform control 0.09 seconds on CPU;
the derangement reused the same image-to-class score matrix.

Private artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-train-tinyclip-control-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`0325251c9e49d2b99368e40e698478f106cd51061512788dea4f54133fbdebd3`;
of `private-predictions.jsonl`:
`7292975ce4b0f9f100aece48566ddb638c3a4cd3de2de74c7ed55eb5bada37da`.
Summary records checkpoint, processor and tokenizer file hashes, task and
page-receipt hashes, protocol and code hashes, timing and all condition metrics.
