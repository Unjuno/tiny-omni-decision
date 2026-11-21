# R2-M RMS-window representation follow-up protocol (2026-11-21)

## Question

Does a 128-bin, per-axis, per-trial-mean-centered RMS envelope improve the
existing frozen audio + vibration Decision task over the current 128-point
linear-interpolation vibration representation, under the already fixed
seven-participant TRAIN-only grouped cross-validation?

This is one representation-path experiment motivated by the
[TRAIN-only temporal-bandwidth audit](R2M_TEMPORAL_BANDWIDTH_AUDIT_20261121.md).
It does not establish that temporal bandwidth caused the earlier fusion result,
and it is not a product or population-quality evaluation.

## Frozen conditions

Only the sensor tokenization changes. The candidate represents each trial by
128 equal-duration bins; within each bin and each of three axes it computes RMS
after removing the full trial's per-axis mean. The resulting `[128, 3]` tensor
keeps the current scorer interface. It is an energy envelope, not a
band-limited waveform resampler and not calibrated physical acceleration.

Keep the existing audio features and the original frozen TRAIN participants,
five-choice-free 24-label vocabulary, seven outer participant folds,
fold-local fit-only sensor normalization, calibration participant rule, seed
17, model sizes, loss, optimizer, 80-epoch schedule, batch size, candidate
ordering, and calibration-macro-NLL checkpoint selector. Compare Audio-only and
Audio+sensor on identical held-out participant IDs and in the existing OOF
order. The primary comparison is paired OOF macro NLL; also report Accuracy,
macro Accuracy, NLL, macro NLL, Brier, macro Brier, ECE, per-participant
differences, and the existing seven-cluster bootstrap diagnostic. No new seeds,
epochs, architecture changes, or post-hoc tuning are permitted in this run.

The feature cache must contain TRAIN rows only. The model CV loader must reject
any feature signal split outside that declared scope. The source split,
pair-index identities, archive receipts, and old audio feature values remain
hash-pinned. The legacy point-sampled cache and run are read-only. The builder
reads only the old cache arrays required to identify TRAIN rows and preserve
their pinned audio features; NPZ compression requires decompressing the
flattened audio feature member, which also contains calibration/validation
spans. Those non-TRAIN spans are never selected, serialized, or passed to the
model. Old sensor feature arrays are not read. Raw calibration/validation
sensor signals are not dereferenced, and sealed-audit signals are not opened.
Existing calibration/validation participant results are not used to tune
this representation.

## Decision rule

The representation is retained for this task only if the paired seven-fold
result improves both macro NLL and macro Accuracy over Audio-only without a
material deterioration in participant-level proper scores. If it does not,
retire this source's audio-vibration fusion path for the supplied-activity
task. Mixed or participant-concentrated effects are inconclusive and do not
justify additional tuning on these rows.

The participant folds are all drawn from the original TRAIN pool and share
overlapping fit participants. This is development evidence only. The exposed
calibration/validation participants and sealed audit remain unopened. No result
here closes R2-M or changes the broader product targets.

## Artifact and execution record

The RMS feature cache and CV output use new external paths; existing caches,
checkpoints, and run artifacts are not overwritten. Record the builder commit,
builder hash, output cache-index hash, effective CV configuration hash, source
archive and feature-cache hashes, start/end time, hardware, elapsed time,
per-fold and pooled metrics, and output file hashes in the results report.
Publish code, protocol, and small result tables only; do not add source media,
feature arrays, or model checkpoints to Git.

The exact command and cache hash are added here after the TRAIN-only cache is
built and integrity-checked, but before the model fit starts.
