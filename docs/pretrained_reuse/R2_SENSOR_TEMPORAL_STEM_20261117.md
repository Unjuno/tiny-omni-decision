# R2-M sensor temporal stem prototype — 2026-11-17

## Implementation

Added `SensorTemporalLSTMStem` in `src/tiny_omni_decision/sensor_temporal_stem.py`.
It consumes already calibrated and training-normalized sensor windows with
shape `[batch, time, channels]`, timestamps in milliseconds, and a boolean
validity mask. For each row it removes invalid observations, appends a
log-scaled elapsed-time feature, and returns the final state of a single-layer,
unidirectional LSTM. This is suitable for causal window encoding and can feed
the existing typed-option readout; it does not itself implement multimodal
fusion or JSON decisions.

The configured 12-channel, hidden-width-64 stem contains **20,224 trainable
parameters**, well below the protocol's provisional 1M budget. The count is
derived from the instantiated PyTorch module, including both LSTM biases.
Invalid padded values/timestamps are ignored; valid timestamps must be finite
and strictly increasing, and every batch row must contain a valid observation.
Channel calibration, normalization, units, clock alignment, and per-channel
missingness remain the caller's responsibility.

## Verification and limits

Added CPU tests for shape/finite output and parameter budget, invariance to
invalid padding (including NaN/Inf padding), finite gradient propagation, and
rejection of empty or non-monotonic valid timelines. These are interface and
numerical tests only. No dataset, media, checkpoint, validation set, sealed
audit, GPU, or cloud compute was used. No model was trained and no accuracy or
temporal-reasoning claim is made.

The quality gate remains open: compare this stem and simple rule baselines on a
rights-approved, source-grouped sensor task with audited labels and known
pretraining overlap before fitting or product use. This prototype does not
close R2-M, the 100–200M full-model budget, 95–98% quality targets, or any
quantization/release gate.
