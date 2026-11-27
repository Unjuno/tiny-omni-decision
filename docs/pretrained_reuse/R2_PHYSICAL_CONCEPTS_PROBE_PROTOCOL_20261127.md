# Physical Concepts one-time probe protocol (2026-11-27)

## Freeze point

This protocol is frozen after selecting the seed-17 VideoPrism-B + 64-wide
LSTM checkpoint by minimum loss on the two freeform training shards and their
independent shard-held-out validation set. The checkpoint SHA-256 is
`40e9fd86aeb384f30459c346e3a5288cc6a7281ccf1e68854b9001bf44af8e49` (selected
epoch 66). Its validation cosine loss is 0.0026964 versus 0.0080918 for
frame-feature persistence. No probe payload had been accessed when this
protocol and its object-generation manifest were committed.

The selected candidate is frozen. Probe results will not select a checkpoint,
change the feature extractor or predictor, calibrate probabilities, or trigger
another probe-guided training run.

## Source and sample

Use the official Google DeepMind Physical Concepts repository at revision
[`baa2776017ea3cc8c873ce682bb6646860e12cae`](https://github.com/google-deepmind/physical_concepts/tree/baa2776017ea3cc8c873ce682bb6646860e12cae)
and its CC BY 4.0 dataset materials. The publisher describes five probe
categories, each containing 5,000 quadruplets of two possible and two
impossible 15-frame videos. See the pinned [dataset notebook](https://github.com/google-deepmind/physical_concepts/blob/baa2776017ea3cc8c873ce682bb6646860e12cae/load_physical_concepts_dataset.ipynb).

For a bounded initial transfer diagnostic, select exactly 512 complete
quadruplets from each category (2,560 total). Each selected record is one
statistical group; the four clips inside it are never split. Select by sorting
all records by the lexicographic SHA-256 rank defined in
[`physical-concepts-probe-512-eval-20261127.json`](../../manifests/pretrained_reuse/physical-concepts-probe-512-eval-20261127.json),
then taking the first 512 per category. The rank uses category, seed 17, and
the record content hash only. It does not inspect labels, model scores, or
predictions. The official 20 shard generations, sizes, and MD5 values for all
five categories are pinned in that manifest; every downloaded shard will also
receive a computed SHA-256 and a TFRecord CRC/schema audit.

This is a predeclared exploratory subset, not a replacement for the complete
publisher probe and not a 95–98% product claim. It covers 10.24% of each
category. The full probe would require 100,000 encoder inputs; the selected
subset requires 10,240. At the measured CPU rate this subset is projected to
take about 3.4 hours for feature extraction. No probe result may be used to
change the candidate or sampling rule.

## Frozen scoring

- Use the already pinned VideoPrism-B encoder, eight uniformly sampled frames
  `[0,2,4,6,8,10,12,14]`, and the selected LSTM checkpoint.
- For each clip, score the mean cosine prediction error over its seven next
  feature transitions.
- For each quadruplet, compare the mean score of its two impossible clips with
  the mean score of its two possible clips. The primary group result is correct
  when impossible has the larger surprise score; ties count as 0.5.
- Also report clip-level threshold results using the 95th percentile of
  per-sequence errors on the already frozen freeform validation split. Compute
  that threshold before opening probe payloads. This is an anomaly threshold,
  not a learned probability calibration.
- For diagnostic per-clip probabilities, let `q95` be that validation
  threshold and `s = 1.4826 * median(abs(validation_error - median_error))`.
  If `s` is zero, use validation-error standard deviation; fail if that is also
  zero. Freeze `p(impossible) = sigmoid((score - q95) / s)`. This maps the
  one-class validation distribution to an anomaly score; it is not calibrated
  on impossible examples.
- Report accuracy, sensitivity, specificity, NLL, Brier, and 10-bin ECE, with
  calibration metrics explicitly labelled diagnostic and uncalibrated.
  Compute uncertainty by paired bootstrap over whole quadruplets, seed 17,
  10,000 resamples.
- Save every sample ID, source generation, possible/impossible clip score,
  prediction, fixed-threshold probability, and group metric. Report each of the
  five categories separately and as an unweighted macro average.

The freeform training videos have moving cameras while the publisher says the
probe videos use static cameras. Results therefore measure transfer across
this source's camera/domain shift, not broad video understanding. Exact overlap
with freeform rows will be checked and recorded. No other model, branch,
modality, or sealed audit is evaluated by this protocol.

## Runtime estimate and artifacts

The measured steady CPU rate on train and validation shards is approximately
1.18 seconds per video. A 10,240-video subset projects to about 12,100 seconds
(3.4 hours), plus download, parsing, and serialization. It runs on the existing
local CPU; no GPU, cloud compute, or paid service is used. Source payloads,
features, and predictions stay under a new directory in
`C:/CodexArtifacts/pretrained-reuse-r2-local/`. Only compact manifests,
reports, and tests are committed.
