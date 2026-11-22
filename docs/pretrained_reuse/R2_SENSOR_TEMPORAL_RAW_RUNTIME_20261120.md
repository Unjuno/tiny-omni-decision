# R2-M raw sensor temporal Decision runtime — 2026-11-20

## Runtime contract

Added `decide_raw_sensor_observation` in
`src/tiny_omni_decision/sensor_temporal_runtime.py`. It accepts one validated,
text-free sensor `ObservationBatch`, a `SensorTemporalDecisionScorer`, a fixed
typed-option schema, expected channel names, training-derived normalization
statistics, and a query time/freshness bound. The referenced NPZ must contain
exactly one `values` array whose time/channel shape matches the observation
metadata. Its bytes are SHA-256 checked before decoding with `allow_pickle=False`.

The adapter requires exact model/channel-name ordering, applies each channel's
declared affine calibration, checks the calibrated values against any declared
range, applies caller-supplied mean/scale, and feeds only valid rows whose
timestamps are at or before the query time to the causal LSTM. If no past
sample exists or the most recent sample exceeds the allowed age, it returns a
typed `insufficient_evidence` abstention with null option probabilities.
Candidate IDs and labels are not model inputs. The caller remains responsible
for fitting normalization on training data only, assigning trustworthy units
and clocks, and defining task policy.

## Verification and limits

Four focused raw-runtime CPU tests verify calibration/normalization parity,
exclusion of future samples, stale/empty evidence abstention, payload-hash and
channel-order rejection, invalid-step handling, and declared-range enforcement.
All 13 sensor temporal tests passed; the full suite passed **2,934 tests with
11 skipped**; `ruff check .` and `git diff --check` passed. The fixtures are
mechanics only. No public dataset, sensor labels, checkpoint, validation,
sealed audit, training, GPU or cloud compute was used. This closes a raw-input
interface gap but does not establish sensor quality or authorize use of any
source.
