# R2-M sensor temporal Decision integration — 2026-11-20

## Change

`SensorTemporalDecisionScorer` composes the causal `SensorTemporalLSTMStem`
with the existing `NativeFeatureDecisionScorer`. It accepts a sensor window,
timestamps, validity mask and caller-owned numeric/enum option features. Its
`decide` adapter returns probabilities keyed by supplied option IDs in caller
order. The adapter does not feed option IDs, class labels or free text into the
network. Candidate reordering permutes scores exactly; it does not change the
score associated with a candidate.

For 12 sensor channels, 8 typed option attributes, LSTM hidden size 64 and
readout hidden size 128, the complete path has **29,697 parameters** (20,224 in
the temporal stem). This remains far below the 1M sensor-path engineering
hypothesis and is additional to—not a replacement for—the resident multimodal
model budget.

## Verification and limits

CPU tests cover the integrated forward path, candidate permutation
equivariance, typed option-ID JSON ordering, probability normalization, and
parameter accounting: **9 focused tests passed**. The full project suite passed
**2,930 tests with 11 skipped**; `ruff check .` and `git diff --check` passed.

This implementation is randomly initialized until fitted. No sensor task,
dataset, weights, validation data, sealed audit or GPU was used. It does not
demonstrate sensor accuracy, calibration, robustness, or superiority over the
existing static HARTH head or a deterministic baseline. Input units,
calibration, training-only normalization, stream freshness and abstention
remain upstream responsibilities. A rights-approved source with audited labels,
independent subject/session splits and verified pretraining-overlap status is
still required before a quality experiment.
