# Shared text features: matched video comparison

Purpose: test whether the same frozen Cross-Encoder already used for text and
image can replace MiniLM in the existing video readout. This is a combined-budget
integration test; it cannot turn near-chance CLEVRER evidence into a general
four-modality quality claim. Keep previous image/video models and artifacts.

Use the same fixed control checkpoint as the image comparison: step1664, SHA-256
0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882.
No numeric-delta model selection or source changes based on these outcomes.

## Fixed pair and controls

- MiniLM-L3 versus frozen Cross-Encoder base-model features, same prompt formatting,
  128-token cap, attended-token mean pooling and 1536-dimensional question/option
  interaction features. The Cross-Encoder is one model, not a copied second tower.
- For EACH text encoder run attention video readout and separately trained no-video
  readout, then evaluate the fixed scene-shuffle control on its selected weights.
- Existing VideoTokenReadout parameters/shape, seed17, 24 epochs, effective batch64,
  microbatch8, AdamW1e-3/wd1e-3, existing normalized source/task weights and loss.
- Capture the baseline head initialization RNG after its existing text extraction
  path, restore it for all four runs, and verify exact head tensor/order hashes.
- Same 3794 train choices/256 scenes and 3014 validation choices/200 scenes.
  Preserve split, asset, question and normalized-content gates. No final audit.
- Same 8-frame VideoPrism grid (2048x768 tokens), same CPU decoder/processor/revision.
  Re-extract once into a transient RAM bank because the old bank was not persisted.
  Require each full-run asset token hash to match the previous saved extraction
  record in addition to pooled-feature parity. Do not write a multi-GB disk cache.
- Select earliest minimum validation macro question-type NLL; best/final separate,
  exact reload, complete train/validation curves and per-question-type metrics.
- Save predictions and scene-cluster paired bootstrap (10000,seed17) for feature
  policy, video-minus-no-video and matched-minus-shuffled comparisons. Report
  Accuracy/NLL/Brier/ECE; no favorable-subset or frame/wording search.

## Gates and interpretation

First validate CPU contracts, then separate small GPU/CPU-worker smoke in a new
output directory. Full run only after smoke passes. No concurrent GPU training;
check RAM bank plus8GiB reserve and output allowance plus512MiB disk reserve.
No new download, encoder updates, cloud use, extra seeds or automatic extension.

A result matching a weak baseline does not establish useful video reasoning.
Do not remove MiniLM from the current runtime or adopt sharing solely for size.
Record all per-type regressions, uncertainty, initial/full/selected accounting,
actual resident parameter counts, wall time and VRAM. The end product remains
<=200M BEFORE quantization and >=95% for each required modality, not this probe.
