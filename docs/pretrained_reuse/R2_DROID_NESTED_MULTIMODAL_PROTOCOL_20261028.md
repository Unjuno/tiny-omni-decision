# DROID nested instruction and video probe

Before execution, reuse the 275 train recordings, sensor data, preceding-step
instruction text and instruction-connected five outer folds of the earlier
probe. Use only their first 275 cached VideoPrism feature rows. Pool the three
views, then average the first and last four temporal feature vectors separately
to obtain 1536 video values. Compare four linear readouts on identical rows:
sensor, sensor + video, sensor + frozen MiniLM CLS, and all three inputs.

For each outer training set, assign its instruction-connected components to
two deterministic seed-17 inner folds. Fit feature normalization only on each
inner fit, and compare updates 1, 10, 50, 100 and 200 using pooled inner NLL.
Choose the lowest NLL per arm (ties choose earlier), refit on the full outer
train set for that many updates and predict the outer fold. Use full-batch Adam,
learning rate .01, weight decay .001, seed 17 and seven natural-prevalence
classes, as in the preceding probe. Normalize sensor and video per feature;
keep pretrained text values unchanged, with a zero vector and missing flag for
missing text. Clamp standardized sensor/video features to [-10, 10]. No reuse of
outer predictions to tune the procedure, no validation/audit scoring, no product
promotion. Fail if any training fold lacks one of seven classes. Store component
assignment, inner scores, selected updates, OOF probabilities and hashes.

This small train-only diagnostic is based on previously examined source and
folds, so its outer predictions are development estimates, not blind evidence.
Early stopping cannot by itself prove multimodal complementarity. Compare
against single-input readouts and retain missing-input and parameter-count
limitations. No new encoder training, online download or paid compute is used.
