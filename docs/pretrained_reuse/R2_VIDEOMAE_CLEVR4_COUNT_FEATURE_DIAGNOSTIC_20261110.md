# VideoMAE Clevr-4 count-feature diagnostic — 2026-11-10

## Result

A fixed multiclass ridge readout was applied to the already-extracted frozen
384-D global VideoMAE features. On five deterministic, class-stratified folds
within the 72,010 training images, out-of-fold Accuracy was **34.77%** and macro
recall was **34.47%** (10-way chance Accuracy is 10%). The probe was fit only
on the other four training folds in each pass. This shows some count signal in
the pooled features, but the fixed linear readout separates it poorly.

The same fixed readout, fit on all training images, scored **30.00% Accuracy**
on the existing 1,000-image selector. The saved selected MLP scored **42.60%**
on those same IDs. The paired image-bootstrap difference (ridge minus MLP) was
−12.6 percentage points, percentile 95% interval [−16.0, −9.2], using 10,000
resamples and seed 17. This is a descriptive post-selection comparison only:
the MLP was selected using this selector, so the interval is not confirmatory
evidence of generalization and was not used for any selection.

Both probes recognized the extreme counts much better than the middle counts.
The train-fold out-of-fold ridge recalls ranged from 99.25% for count 1 and
88.26% for count 10 down to 8.89% for count 9. On the selector, ridge recall
was 100% for count 1 and 90.98% for count 10, but 5.32% for count 9 and
11.32–28.89% for counts 2–8. This is consistent with a weak count signal in
the global feature and a substantial readout/representation limitation for
intermediate counts; it does not isolate which is the main cause.

| Count | Train-fold OOF support | OOF ridge recall | Selector support | Selector ridge recall | Existing MLP selector recall |
|---:|---:|---:|---:|---:|---:|
| 1 | 7,484 | 99.25% | 39 | 100.00% | 97.44% |
| 2 | 7,281 | 23.88% | 90 | 28.89% | 73.33% |
| 3 | 7,251 | 29.73% | 107 | 26.17% | 66.36% |
| 4 | 7,514 | 23.44% | 101 | 25.74% | 41.58% |
| 5 | 7,130 | 14.00% | 106 | 11.32% | 24.53% |
| 6 | 7,172 | 22.27% | 123 | 19.51% | 46.34% |
| 7 | 7,182 | 21.08% | 123 | 13.82% | 8.94% |
| 8 | 7,094 | 13.91% | 95 | 12.63% | 27.37% |
| 9 | 6,815 | 8.89% | 94 | 5.32% | 5.32% |
| 10 | 7,087 | 88.26% | 122 | 90.98% | 68.85% |

## Frozen procedure and integrity

The protocol was committed before the diagnostic ran in
[`R2_VIDEOMAE_CLEVR4_COUNT_FEATURE_DIAGNOSTIC_PROTOCOL_20261110.md`](R2_VIDEOMAE_CLEVR4_COUNT_FEATURE_DIAGNOSTIC_PROTOCOL_20261110.md).
The fit was one-hot least-squares ridge with fixed relative coefficient 0.01,
fold-local standardization, and deterministic within-class folds assigned by
SHA-256 of the image ID. No hyperparameters were selected, no probability or
calibration claim was made, and the model/encoder weights were unchanged.

Only training and selector labels were read. The compressed feature archive
was streamed through the 72,010 train and 1,000 selector rows; report feature
rows were not decoded into the analysis array. Reporting labels and
predictions, official validation, and sealed audit were not read. The frozen
cache, input receipt, split manifests, and saved selector predictions all
matched their registered SHA-256 values. Receipt and feature hashes, fold
metrics, confusion counts, runtime, runner commit and runner hash are in the
external machine-readable result:

`D:\CodexArtifacts\pretrained-reuse-r2\videomae-clevr4-count-feature-diagnostic-20261110-attempt-02\diagnostic.json`

Result JSON SHA-256:
`55c4eb4096b474fd46b87112bef3601b86dbdd699d2e28cef21bad7005ed4814`.
The CPU diagnostic took 31.44 seconds. No GPU, model inference, data download,
or paid compute was used. The first invocation stopped before fitting because
NumPy 2.4 renamed an internal header helper; the code was corrected to use the
public v1/v2 header readers and the successful run used a separate attempt-02
directory. The original features and Path A experiment artifacts were not
changed.

## Interpretation and next step

This does not make the image task pass: the original frozen MLP's one-time
reporting result remains 43.43% count Accuracy, and that reporting result was
not reopened here. The diagnostic suggests a global pooled feature plus a
small readout is a poor route to reliable object counting; it neither proves
that all VideoMAE features lack count information nor justifies choosing a new
architecture from the exposed selector. Do not tune the existing checkpoint
on the report pool. A future count-focused study would need a separately
frozen protocol and new untouched data, likely preserving spatial/token or
object-centric evidence. Product-level multimodal quality remains unmet.
