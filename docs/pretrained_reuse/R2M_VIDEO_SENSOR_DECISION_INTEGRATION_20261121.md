# Video + sensor temporal Decision integration (2026-11-21)

## What changed

Added `VideoSensorTemporalDecisionScorer`, a small trainable readout that joins
caller-supplied frozen per-frame video features with a causal `SensorTemporalLSTMStem`.
The sensor window is encoded into one state token timestamped at its final valid
sample. `TimeAwareGatedFusion` conditions on caller-supplied context and typed
candidate vectors, so the same observation can be queried under multiple
decision conditions. `FrozenFeatureCandidateScorer` returns one masked logit
per supplied choice.
The inference adapter emits the repository's fixed-taxonomy probability JSON and
preserves caller option IDs and order.

Both streams may be independently absent, but at least one must be present for
each example. Presence is separate from numeric zero. Frozen encoders, feature
normalization/calibration, clock synchronization, option construction, abstention,
and data-specific training remain outside this module. This primitive does not
load raw media or include encoder parameters in its parameter count.

## Verification

`uv run pytest tests/test_video_sensor_temporal_decision.py -q`: **6 passed**.
`uv run ruff check src/tiny_omni_decision/video_sensor_temporal_decision.py
tests/test_video_sensor_temporal_decision.py`: **passed**.

The hash-checked runtime adapter is implemented in
`src/tiny_omni_decision/temporal_video_sensor_runtime.py`. It verifies complete
NPZ payload hashes and exact timestamps/masks, applies declared sensor-channel
calibration followed by supplied train-derived normalization, masks future or
stale rows, abstains on required stale streams, and then calls the temporal
scorer. Text criteria can provide a question vector only through the persistent
observation feature cache: its key must identify the exact criterion text, the
frozen encoder/preprocessor revisions and the expected vector width. The cache
payload digest is rechecked before use. Spoken criteria and text parameters that
are not represented by that vector fail closed. Seven focused runtime tests
cover JSON order, calibration, stale required/optional streams, future evidence,
clock mismatch, hash corruption and mismatched text-question cache keys. The
entire CPU suite after text-cache support: **3,004 passed / 11 skipped**; Ruff
and `git diff --check` pass.

Tests exercise distinct decision contexts on the same observation, candidate
permutation, option masking, video-only/sensor-only and
both-missing behavior, present-zero versus missing sensor, trainable paths through
the sensor LSTM/fusion/readout, JSON option ordering, and strict configuration
checkpoint round-trip. These are CPU mechanics tests only. The model is randomly
initialized and untrained; no dataset, quality improvement, multimodal benefit,
calibration, or product admission is established. No training data, checkpoint,
or sealed evaluation was used.

For an explicit size example, a configuration with 768-wide video tokens,
6 sensor channels, 128 context features, 64 option features, 64-wide sensor
state, and 96-wide fusion totals **295,235 trainable readout parameters**:
18,688 sensor LSTM, 239,426 fusion, and 37,121 candidate scorer. Adding the
already measured 114,365,184-parameter frozen VideoPrism-B gives 114,660,419
parameters for that visual-plus-sensor path only. This excludes audio/text
encoders and is not a complete Omni parameter or package-size claim.

A CUDA mechanics smoke on the local RTX 3080 Laptop used synthetic supplied
features with batch 1, 8 video tokens, a 250-step sensor window, and 21 choices.
The output was finite `[1,21]`; 20 warmed-up model-only forwards averaged
17.79 ms. Peak PyTorch allocation/reservation was 18.57/25.17 MB. Environment:
PyTorch 2.6.0+cu124, CUDA 12.4. This excludes video/audio feature extraction,
preprocessing, host/device transfer and application overhead; it is not an
end-to-end latency or quality measurement.

The full hash-check/load/normalize/Decision JSON path was also smoke-tested
with a temporary synthetic cache on the RTX 3080: 8 × 768 video features,
250 × 6 sensor rows, 128 context features and 21 choices. Ten runs averaged
17.17 ms including file reads, SHA-256 checks, calibration, CPU→GPU transfer
and inference; peak PyTorch allocation was 18.57 MB. The probability sum was
1.00000002 and status was `selected`. The temporary cache was removed at process
exit. This validates execution plumbing only; the features and model weights
were synthetic/random, so it provides no task-quality evidence.

## Next evidence gate

Before fitting, admit a rights-cleared paired video+sensor source with a stable
participant/scene split and task labels that require the supplied options. Freeze
the split and preprocessing, then compare video-only, sensor-only, and paired
conditions under the same examples. Include missing-stream, shuffled-pair,
time-offset, and metadata-only controls. Do not interpret this integration as
support for a fusion gain until those controls and validation results exist.
