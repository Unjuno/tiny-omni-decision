# VideoMAE Clevr-4 count-feature diagnostic — preregistered protocol

## Question

Does the frozen 384-dimensional global VideoMAE feature contain a stable,
linearly decodable count signal on training examples, or does count remain
poor even for a fixed low-capacity readout? This is a diagnosis of one narrow
synthetic image task, not a new quality claim or a candidate-selection run.

## Fixed procedure

- Source features are the existing immutable VideoMAE 100k feature cache and
  frozen Path A input receipt from the 2026-11-10 run. No encoder inference,
  model update, hyperparameter search, calibration, or checkpoint selection is
  performed.
- Read only the 72,010 training labels and the 1,000 selector labels. Open the
  feature archive only through the first 73,010 rows (train then selector),
  verifying the prefix against its feature manifest. Do not read reporting
  labels, reporting predictions, official validation, or sealed-audit data.
- Fit an unweighted multiclass ridge readout on count labels only. Use five
  deterministic stratified folds over the 72,010 training images, assigning
  within each class by the order of
  `SHA256("seed17:clevr4-count-feature-oof-v1:" + image_id)`. In every fold,
  standardize dimensions using that fold's fit portion only, center one-hot
  targets, and solve the normal equations with normalized Gram matrix and
  fixed relative ridge coefficient `0.01`. No value is selected from results.
- Separately fit the same fixed recipe on all training images and score the
  already-exposed 1,000-image selector once. Compare descriptively with the
  saved epoch-8 MLP predictions on the exact same selector IDs. A paired
  10,000-resample image bootstrap uses seed 17; because the MLP checkpoint was
  selected on this selector, this comparison is post-selection and is not
  inferential evidence of generalization.
- Report accuracy, macro recall, per-class recall/support and confusion counts.
  Ridge scores are not calibrated probabilities; do not report NLL, Brier, or
  ECE for them. The original MLP probability metrics remain unchanged.

## Interpretation limits

The five-fold result is internal train-only cross-validation of a diagnostic
readout on a frozen representation. It can indicate whether a simple linear
boundary recovers count information from pooled features; it cannot establish
that a larger model, token-level features, or another dataset would improve
the product. Selector comparisons are descriptive only. The one-time reporting
pool remains closed to this diagnostic.
