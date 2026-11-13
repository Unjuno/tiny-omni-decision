# Training-only physical mirror diagnostic

Freeze the spatial/bilinear epoch16 checkpoint. Use first64 training images in
pinned manifest order; no validation/sealed audit forward. Re-extract original
and physically horizontally mirrored images through the unchanged frozen DINO
224x224 whole-image preprocessing, batches32. Compare identity, physical mirror,
and horizontal cell reversal of original features. Four tasks/all40 options,
unchanged labels. This is training-set robustness, not generalization.

Require original re-extraction parity with cache <=1e-5 before scoring. Preserve
hashes, features, per-example predictions and metrics. No training, calibration,
selection, ensembling or follow-up transformation search. New output directory.
