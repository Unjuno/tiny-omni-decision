# Fixed native full-field-of-view comparison

This bounded experiment follows the pipeline audit and failed positive-support
policy. Preserve all historical runs. Use the original native32,768-update
control, not the failed STARSS policy. The user's autonomous local research
authorization applies; no paid compute or intervention in existing processes.

## Single intervention

For Image and Video, replace resize-short-side/center-crop with one fixed policy:
resize the full RGB extent to fit288x288, bilinear interpolation, centered black
padding, float32/255. Preserve aspect ratio subject to integer rounding. No pad
mask or new tokens: the existing readout sees the same token layout.

Image640x480 becomes288x216 plus36px top/bottom padding. Video480x320 becomes
288x192 plus48px padding. This preserves full spatial extent while reducing object
scale and adding padding; it is not a pure isolated estimate of removed-pixel
information and does not preserve every original pixel at its original resolution.
Do not optimize padding color, resolution, crop variants or seeds against results.

Same pinned VideoPrism114365184-parameter checkpoint/source, frozen encoders,
single-frame Image spatial17x768 features, eight-frame Video2048x768 tokens,
frame indices/decoder/order, coordinates/masks, all source IDs/splits/questions,
candidate queries, original sampling order and option permutation rule.
Audio/Text feature banks remain identical. No recurrent architecture, new frames,
encoder tuning, calibration or sealed audit access in this comparison.

## Stages and protection

1. Pure CPU tests for resize geometry, corner retention, padding and input dtype.
2. Separate real JAX/TF CPU smoke: first8 TRAIN images and first2 TRAIN scenes.
   Recompute the original center-crop features and require exact equality to the
   saved original arrays. New features must be finite, shape-compatible, actually
   changed, and bit-exact on repeated encoding/save/reload. Record pinned versions,
   parameter count, wall time, memory, input geometry and hashes. No quality metric.
3. Only after a passing practical smoke, create all8936 Image and456 Video arrays
   in a fresh directory. Check each media hash, original observation-to-bank index,
   exact smoke/full output parity, complete counts, source hashes and file reloads.
   Resource ceilings:5400s,6GiB process peak working set,5GiB disk reserve. Exceeding
   a ceiling means incomplete extraction, not a quality failure. No silent restart.
4. Bind only those458 new visual banks to a separate effective config, preserving
   frozen corpus/queries and every nonvisual bank. Verify shapes/indexes and split
   identities before a separate training smoke and one full seed17 fit.

Only local compute; existing JAX0.6.2/TF2.18.1 environment first, no upgrades.
Immutable new feature/config/output paths. Do not edit executing source. Record
the exact source bytes and config/input/artifact hashes. No historical cache,
checkpoint, corpus, Teacher/QAT artifact or user worktree is modified.

## Fixed training and evaluation

Original readout initialization, fresh AdamW,889345 trainable parameters and
196254874 total prequantization parameters. Same6656 unique questions,262144
presentations,32768 updates,8 microbatches/update, LR.001/warmup62/cosine32768,
weight decay.01,clip1,CE and no performance early stopping. Same evaluation steps
0,512,1024,2048,4096,8192,12288,16384,24576,32768. Same full24346 validation IDs,
order and options; keep all failures. Choose best using unchanged hierarchical
task/source/modality macro NLL, min_delta1e-5, earliest tie, initial0 eligible.
Save best and final separately and require fresh-process exact replay.

Primary comparison is selected versus original-control selected8192; also compare
equal8192 and32768 and full curves. Report all modality/source/task Accuracy/NLL/
Brier/ECE, macro/minimum, temporal/static Video, unique IDs/assets, TRAIN gaps,
LR/gradients/VRAM/wall time. Full TRAIN mixtures differ from validation; retain
common-task analysis and avoid interpreting rolling optimization as a fixed gap.
Initial Text/Audio predictions must equal the control; initial visual predictions
may change because preprocessing is the intervention.

Retain for further research only if selected macro NLL improves, mean Image/Video
Accuracy improves, and no modality declines more than2pp. This is an engineering
flag, not product promotion or a weaker95–98% gate. Report smaller trade-offs too.
Use paired image/scene-cluster bootstrap5000,seed17,linear percentile95% for visual
Accuracy/NLL/Brier differences; acknowledge development/selection bias. If it fails,
preserve the result and do not tune geometry against validation. The full Omni
quality, joint/temporal/OOD, calibration and deployment goal remains unchanged.
