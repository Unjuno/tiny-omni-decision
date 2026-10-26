# Synthetic AV temporal alignment contract

## Intent and scope

Make the synthetic audio/video sequence task's temporal input contract explicit
and mechanically check that rendering preserves it. This is a pipeline and
architecture diagnostic only. The generated task does not measure natural
audio/video understanding or the product's 95–98% quality target.

## Temporal representation

- Each sequence has exactly 8 ordered steps.
- A video step is one 224×224 rendered frame at 2 frames per second.
- An audio step is one 0.5-second mono tone segment at 32 kHz (16,000 PCM
  samples).
- Video frame `t` and audio segment `t` describe the same symbol and start at
  the same time, `t × 0.5 seconds`.
- Frozen VideoPrism features have shape `[batch, 8, 768]`, one framewise pooled
  feature per decoded frame. Frozen EfficientAT features have shape
  `[batch, 8, 384]`, one embedding per tone segment. Encoders remain frozen.
- The ordered decoder output and waveform segment order are checked against
  the generated manifest. No temporal pooling or interpolation is allowed in
  the input contract.

## Readout comparison

The reference is the direct two-layer temporal LSTM over raw per-step audio,
video, and query features. The most interpretable gated candidate is the
identity-initialized residual gate: the reference input remains present and
the gate scales each raw modality by a bounded factor in `[0.5, 1.5]`. At
initialization it produces exactly the reference logits. This prevents a
softmax gate from erasing a stream before the recurrent layer can learn from
it.

The existing experiments also tested query-conditioned bidirectional
cross-attention, a query-conditioned temporal gate, and a gated path with
projected-stream bypasses. Their code and negative results remain preserved in the
existing reports; none is retrained or selected again from the exposed
development validation.

## Data and interpretation limits

- Seed is 17. Train and validation are disjoint by group, full sequence
  identity, and complete generated media asset.
- The task reuses four visual glyph and four tone prototypes across splits;
  only complete sequences/pairs are held out. It therefore tests temporal
  order binding, not unseen-symbol or natural-media transfer.
- The existing validation was used for checkpoint selection. It is exposed
  development data, not a blind audit, and receives no additional model
  selection in this clarification.
- No sealed audit, new fitting, checkpoint change, or GPU run is part of this
  input-contract update.

## Verification

The test suite uses the same video decoder as feature extraction and checks
each decoded frame against its source frame in order. It also checks decoded
video presentation timestamps and duration, exact PCM sample count, tone
frequency and order, and audio energy near each segment's boundaries. This
verifies the renderer/decoder contract; it does not validate VideoPrism or
EfficientAT representation quality.

On 2026-10-26, the AV task and native interaction contract suite passed 25
tests, and the synthetic readout, training, and feature suite passed 18 tests.
Ruff passed for the changed test. The prior readout reports already retain
separate CPU smoke and checkpoint-reload results; no new model run was needed
for this input-contract test.

