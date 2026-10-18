# Fresh Clevr-4 100k frozen-image readout protocol (2026-10-19)

## Question

Does substantially more unique Clevr-4 training coverage improve the existing
small DINOv2 candidate scorer on a deterministic internal validation split
whose media bytes do not overlap the prior Clevr-4 train manifest?

This is a narrow synthetic image-taxonomy experiment, not a claim about
natural-image understanding, video, other modalities, or a unified product.

## Frozen inputs

- Source: official Clevr-4 100k archive, pinned size and SHA-512 in the GPU
  smoke/acquisition record.
- Source partition: official `train` only; its official `val` partition is
  untouched and is not loaded by this experiment.
- Internal split: seed 17, asset identity, 72,010 train / 8,002 validation.
- Media checks: zero train/validation overlap and zero overlap with all 8,424
  entries in the previously used Clevr-4 train media manifest.
- Image representation: frozen DINOv2-small CLS token concatenated with mean
  patch token, FP32, full-image 224x224 bicubic resize.
- Decision readout: existing `FrozenFeatureCandidateScorer(1152, 128)`;
  768 image features concatenated with frozen MiniLM candidate-option
  embeddings; all ten supplied options retained for each of four taxonomies.

## Fixed fit and selection

- Seed 17; AdamW, learning rate 0.001, weight decay 0.0001.
- Eight epochs, batch of 64 images; all four taxonomy decisions per image.
- No architecture, input, data or optimizer sweep is planned.
- Save validation metrics at every epoch. Select the earliest epoch with the
  lowest unweighted macro NLL across color, texture, count and shape.
- Preserve both final and selected readout states. Validate selected-state
  reload by exact prediction equality.
- Report per-taxonomy Accuracy, NLL, Brier and 15-bin ECE, macro metrics,
  training loss, example counts, model parameter counts, runtime and VRAM.

The 8,002-image internal validation split is development data and participates
in best-epoch selection. It is neither blind nor a sealed audit. No official
validation labels/media, prior audit set, or ternary quantization is used.
