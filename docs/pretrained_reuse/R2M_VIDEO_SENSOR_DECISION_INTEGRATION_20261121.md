# Video + sensor temporal Decision integration (2026-11-21)

## What changed

Added `VideoSensorTemporalDecisionScorer`, a small trainable readout that joins
caller-supplied frozen per-frame video features with a causal `SensorTemporalLSTMStem`.
The sensor window is encoded into one state token timestamped at its final valid
sample. `TimeAwareGatedFusion` conditions on the caller's typed candidate vectors,
and `FrozenFeatureCandidateScorer` returns one masked logit per supplied choice.
The inference adapter emits the repository's fixed-taxonomy probability JSON and
preserves caller option IDs and order.

Both streams may be independently absent, but at least one must be present for
each example. Presence is separate from numeric zero. Frozen encoders, feature
normalization/calibration, clock synchronization, option construction, abstention,
and data-specific training remain outside this module. This primitive does not
load raw media or include encoder parameters in its parameter count.

## Verification

`uv run pytest tests/test_video_sensor_temporal_decision.py -q`: **4 passed**.
`uv run ruff check src/tiny_omni_decision/video_sensor_temporal_decision.py
tests/test_video_sensor_temporal_decision.py`: **passed**.

Tests exercise candidate permutation, option masking, video-only/sensor-only and
both-missing behavior, present-zero versus missing sensor, trainable paths through
the sensor LSTM/fusion/readout, JSON option ordering, and strict configuration
checkpoint round-trip. These are CPU mechanics tests only. The model is randomly
initialized and untrained; no dataset, quality improvement, multimodal benefit,
calibration, hardware performance, or product admission is established. No GPU,
training data, checkpoint, or sealed evaluation was used.

## Next evidence gate

Before fitting, admit a rights-cleared paired video+sensor source with a stable
participant/scene split and task labels that require the supplied options. Freeze
the split and preprocessing, then compare video-only, sensor-only, and paired
conditions under the same examples. Include missing-stream, shuffled-pair,
time-offset, and metadata-only controls. Do not interpret this integration as
support for a fusion gain until those controls and validation results exist.
