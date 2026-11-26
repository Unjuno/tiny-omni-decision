# Screen region pilot: rejected

Source04cc1c0. Fixed30epoch frozen-DINOv2 linear classification completed on the
application-disjoint48screen pilot. Minimum validation NLL selects epoch1.

| Model | Accuracy | Balanced Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Frozen visual linear head |0.334716|0.262416|2.102968|0.808964|0.094643|
| Train-frequency prior |0.486860|0.125000|1.834577|0.809287|0.180519|

Visual features improve class-average recall but do not beat the prior in sample
Accuracy or NLL. Reject this candidate. No95% screen-quality or autonomous-action
claim is supported. Do not tune away the observed weak classes. The prior uses
Laplace smoothing; it receives no validation labels during fitting.

The classifier has6,152 parameters plus22,056,576 frozen encoder parameters.
Feature extraction took7.14seconds; total measured model-load/extraction/fitting
time7.55seconds. Local RTX3080Laptop peak allocated VRAM195,713,536bytes; linear
training on CPU. Selected reload probabilities match exactly. All epochs,
normalization, predictions, metadata and separate best/final weights remain at
`C:/CodexArtifacts/pretrained-reuse-r2/groundcua-classifier-01`.
[Full metrics and hashes](R2M_SCREEN_CLASSIFIER_RESULTS_20261012.json).

Limitations: supplied regions are oracle localization; blank annotations and
invalid boxes lack usable targets and remain explicitly listed in preflight.
Exact decoded crop cross-split overlap is zero. App grouping prevents reuse of
the same app, but shared UI styles and near-duplicate widgets remain possible.
Only4 validation apps; no confidence bound on broad UI understanding is justified.
No calibration or sealed-audit data read. Existing artifacts unchanged.

Small isolated-region DINOv2 features and incomplete coarse labels do not establish
the native screen action task. Preserve this failed prerequisite probe. The next
research decision must address task/representation suitability and genuine joint
evidence; merely increasing epochs on this pilot is not justified.
