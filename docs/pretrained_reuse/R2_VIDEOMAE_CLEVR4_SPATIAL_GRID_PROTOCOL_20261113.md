# VideoMAE Clevr-4 spatial-grid candidate protocol — 2026-11-13

## Hypothesis

The completed global-vector Path A run averages the final 1,568 VideoMAE patch
tokens into one 384-value vector. Its fixed-selector count Accuracy is 42.6%,
while color and shape exceed 99%. This motivates one bounded feature-pooling
candidate: retain coarse spatial information in a 4×4 grid before the same
taxonomy-conditioned MLP. It does not establish that spatial pooling is the
cause of count errors.

## Frozen comparison

- Encoder, checkpoint, preprocessing, Clevr-4 task/labels and parent corpus
  receipt are unchanged from the global-vector Path A run.
- Use seed 17, all 72,010 frozen training images, the same 1,000 selector
  images, eight epochs, batch 512 task examples, AdamW, LR `1e-3`, weight decay
  `1e-4`, constant schedule, and minimum equal-taxonomy selector NLL selection.
- Keep the MLP hidden width (128), taxonomy embedding width (16), task order,
  optimizer and selection rule fixed. Only the representation pooling changes
  to a 4×4 spatial grid; the resulting readout has 789,962 trainable parameters.
- Generate features for train and selector only. Do not extract the previously
  reported 5,296-image partition, load its labels, or use it for candidate
  selection. Do not load sealed audit data.
- This is one predeclared candidate. No grid-size sweep, extra seed, epoch
  extension or hyperparameter search is included. Results on the reused
  selector are development-only and cannot establish an independent gain.

## Feature definition

The pinned VideoMAE ViT-Small K710 encoder remains frozen. Each static image is
repeated over 16 frames. Final tokens are reshaped to temporal tubelets × 14 ×
14 × 384, averaged over temporal tubelets, and adaptive-average-pooled to 4×4.
The pretrained `fc_norm` is applied separately to each grid cell. Flattening
produces 6,144 FP32 values. No temporal evidence is introduced.

The training/selector-only media manifest is frozen separately under
`D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-spatial-inputs-20261113-attempt-01`.
Its receipt records 72,010 train + 1,000 selector rows, 73,010 unique assets,
zero ID/media overlap, and that reporting/sealed data were not loaded. Features
and readout checkpoints are external artifacts and will not be committed.

## Preflight

Six focused CPU tests, full Ruff, and the repository CPU suite pass (2,894
passed, 11 skipped). An actual eight-image CUDA feature smoke emitted finite
`[8, 6144]` features from the pinned checkpoint in 2.53 seconds with 1,234,866,176
bytes peak allocated VRAM. The experiment should proceed to full feature
extraction only with a new output directory and the frozen train/selector-only
manifest. No full extraction or spatial-grid fit is represented by this
protocol.
