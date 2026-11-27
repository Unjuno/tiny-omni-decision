# IntPhys 16-frame Path A diagnostic protocol (2026-11-27)

## Question

Does changing only uniform RGB input coverage from eight to sixteen frames
improve the fixed causal next-feature LSTM's discrimination on the already
observed IntPhys development diagnostic, relative to the eight-frame run and
feature persistence? The preceding label-blind-first data audit found exact
source-frame sequence collisions in 9/90 scene groups at eight frames and
none at 16 frames. This motivates one 16-frame candidate; it does not imply
that the added frames are task-sufficient or improve quality.

This is a **repeated development evaluation**, not a new or blind test. The
official hidden test remains untouched. The result cannot establish product
quality or generalize beyond this task/source.

## Fixed versus changed conditions

The sole intended model-input change is frame count, **8 → 16**. Use the exact
predeclared positions `[1, 7, 14, 20, 27, 34, 40, 47, 53, 60, 67, 73, 80, 86,
93, 100]`. Keep fixed:

- VideoPrism-B model revision `be719a406d563b66f0ac969e7c94bab8e997c81a`,
  source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, frozen.
- IntPhys train archive SHA-256
  `5f4ce9b2240bdbd57e32ed2d8315300314d30572f39adb0d9e58afa3b27da83a`;
  use the exact 3,004 train and 746 validation IDs, ordering, and group
  assignments from the existing seed-17 8-frame split. Fail closed if any ID,
  group, or order differs. Do not re-split or re-sample.
- Causal next-feature LSTM, hidden width 64 (264,960 trainable parameters),
  seed 17, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 64,
  maximum 100 epochs, patience 8, `min_delta=1e-4`; choose the minimum raw
  internal-validation cosine-loss checkpoint. `min_delta` affects patience,
  not best-checkpoint replacement.
- Persistence reference, possible-only validation 95th-percentile thresholds,
  and paired bootstrap over whole scene groups (2,000 replicates, seed 17).
- The 360-video / 90-scene dev sample IDs and order from the existing fixed
  diagnostic. Verify exact ID, group, label, and order parity before scoring.

The previous 8-frame checkpoint remains read-only. This candidate starts from a
new predictor initialization with the same seed and training recipe; it is a
matched re-fit under a changed sequence length, not a strict continuation of
the 8-frame weights. The selected checkpoint and final artifacts use a new
external output directory and cannot overwrite the earlier run.

## Run sequence and gates

1. Add `--frame-count` support while preserving eight-frame defaults. Add a
   pinned one-video, 16-frame CPU smoke. Verify VideoPrism output shape `[1,16,
   2048,768]` before spatial pooling (or its equivalent upstream output),
   upstream token shape `[1,4096,768]`, pooled shape `[1,16,768]`, finiteness,
   and measured peak working set/time.
2. Extract train/validation 16-frame features using pinned source and checkpoint
   receipts. Verify exact baseline sample IDs/order/group assignments and zero
   cross-split full-RGB/status/normalized-input identity overlap. Keep all
   100-frame source checks.
3. Train one seed-17 predictor with the frozen recipe and preserve the best
   validation checkpoint, complete training history, metadata, and hashes.
4. Extract 16-frame features for the same fixed dev IDs and order; verify
   disjoint IDs/groups/model-input hashes before scoring.
5. Run one fixed comparison against the matched 8-frame artifact and
   persistence. Report accuracy, balanced accuracy, AUC, sensitivity,
   specificity and paired scene-group bootstrap. These scores are development
   data reused after prior observation, not blind evidence.

Stop for archive/hash/split mismatch, non-finite features/loss/gradients,
checkpoint reload mismatch, process/runtime failure, or storage/RAM safety
issues. Do not stop early merely because early metrics are poor. Do not repeat
the dev score or change model settings after seeing it. This diagnostic fits
only possible examples; it does not resolve the deeper task-supervision gap.

## Reproducible configuration

Machine-readable frozen configuration:
[`intphys_frame16_patha_seed17.json`](../../configs/pretrained_reuse/intphys_frame16_patha_seed17.json).
Output root: `C:/CodexArtifacts/pretrained-reuse-r2-local/intphys-frame16-patha-20261127-attempt-01/`.
All model, feature, training, prediction, and evaluation artifacts remain
outside Git.
