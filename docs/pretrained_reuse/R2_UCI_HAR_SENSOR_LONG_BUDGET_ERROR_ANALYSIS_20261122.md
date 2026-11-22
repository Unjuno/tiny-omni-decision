# UCI HAR long-budget validation error analysis — 2026-11-22

## Scope and integrity

This post-hoc diagnostic compares the saved best-checkpoint validation
predictions from the 100-epoch UCI HAR static-summary and causal-LSTM runs. It
uses only the already exposed 1,324-row validation split (participants 5, 11,
14, and 30). It does not load a model, checkpoint, source archive, official
test, or sealed audit; it does not fit or select any model. Because validation
NLL selected the checkpoints, this analysis is descriptive and must not be
treated as an independent quality estimate.

Before computing metrics, the runner checked the candidate-freeze and artifact
index, the split identity, train/validation participant disjointness, exact
validation sample-ID order, and the indexed byte counts and SHA-256 hashes for
both saved prediction files. For each candidate, the best-checkpoint reload
prediction file is byte-identical to the original saved best validation
prediction file. `official_test_split_loaded` is explicitly false in the
frozen split receipt. The one-time public-test result remains closed.

The analysis implementation was added on top of source commit
`e23f85b6872bddc662cf9a5759f76d85bcb3c8ee`; the prior sensor fit itself was
produced at `a43662b0af0d685d02aa4460ad259664d401bc2f`.

## Overall paired result

| Candidate | Best epoch | Parameters | Accuracy | Macro Accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static summaries | 84 | 4,097 | 76.06% | 76.39% | 0.6270 | 0.3464 | 0.0521 |
| Causal LSTM stem | 15 | 27,905 | 62.01% | 62.39% | 0.7278 | 0.4232 | 0.0294 |

Of 1,324 paired windows, both candidates were correct on 684, both wrong on
180, only static was correct on 323, and only LSTM was correct on 137. Static
accuracy exceeded LSTM by 14.05 percentage points on this fixed validation
split. Lower LSTM ECE did not offset its lower Accuracy and worse NLL/Brier.

## Class and participant diagnostics

| Activity | n | Static recall | LSTM recall | Most common static → LSTM transition for this true class |
|---|---:|---:|---:|---|
| WALKING | 239 | 73.22% | 74.90% | WALKING → WALKING: 158 |
| WALKING_UPSTAIRS | 220 | 88.64% | 100.00% | WALKING_UPSTAIRS → WALKING_UPSTAIRS: 195 |
| WALKING_DOWNSTAIRS | 200 | 82.00% | 96.00% | WALKING_DOWNSTAIRS → WALKING_DOWNSTAIRS: 162 |
| SITTING | 213 | 85.45% | 0.00% | SITTING → STANDING: 178 |
| STANDING | 222 | 72.52% | 99.55% | STANDING → STANDING: 160 |
| LAYING | 230 | 56.52% | 3.91% | LAYING → STANDING: 120 |

The LSTM predicted `STANDING` for 209/213 true `SITTING` windows and 220/230
true `LAYING` windows. Its high `STANDING` recall (221/222) therefore masks a
severe stationary-class collapse. The static candidate was more balanced on
these stationary classes, though its total accuracy remained below the
project's aspiration.

| Held-out participant | n | Static Accuracy | LSTM Accuracy | LSTM − static |
|---:|---:|---:|---:|---:|
| 5 | 302 | 74.17% | 67.88% | −6.29 pp |
| 11 | 316 | 81.33% | 66.46% | −14.87 pp |
| 14 | 323 | 64.40% | 47.99% | −16.41 pp |
| 30 | 383 | 83.03% | 65.54% | −17.49 pp |

Static had higher accuracy on all four held-out participants. Four groups are
too few for a useful participant bootstrap interval; this report shows each
participant result and makes no interval or population-level claim.

## Interpretation and next action

For this prewindowed six-axis UCI HAR task, the saved LSTM candidate has a
consistent participant-level accuracy deficit and collapses `SITTING` and
`LAYING` into `STANDING`. More epochs did not fix the failure in the preceding
100-epoch run. This does not prove that recurrent models generally fail, nor
does it justify an xLSTM/capacity search. The next sensor-side engineering
work should inspect the existing input/normalization/window pipeline and the
stationary-class prediction behavior before proposing a new fit. Keep this
sensor diagnostic separate from the four-modality objective: UCI HAR is
single-modality, sensor is optional, and this result does not advance the open
natural audio/video joint-input gate.

## Reproduction

- Command:
  `PYTHONPATH=src python scripts/analyze_uci_har_validation_predictions.py --run-dir C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100 --output C:\CodexLocalArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\validation-error-analysis-20261122-attempt-01\result.json`
- Source validation run: `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100`.
- Output JSON: `C:\CodexLocalArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\validation-error-analysis-20261122-attempt-01\result.json`.
- Output SHA-256: `248cc6c88d5ae5a5497287c5b1058b3bab8c1b7328638c0f8454f8ef4c79e79c`.
- Analysis module SHA-256: `886764162a1d081281080d625347e4ccc449040b7987020438c9d4a33495675b`.
- Runner SHA-256: `6117b61232d64a7fb8be1085f47e798c4c8166e55770e8f0f2f5e3072f968ba1`.
- Focused test SHA-256: `5037eafcb8d87997f441db8f5d265bd5ded31e82527a28835bd21f96bfdc8c52`.
- Candidate-freeze SHA-256: `f616fced31d61a4cf16b5372e4a7e61439f401b16123c2c97cf2b4aa5994f8fe`.
- Artifact-index SHA-256: `ec3cf7e471989e48af498644592ec4b3315257a5495721038c5e6c98ae787caa`.
- Split-manifest internal identity: `3e70b00bf16bd5eed7d3cfb04dfe66f7de78029f0c600d34c034584c5f45884c`; file SHA-256: `fcaf7100e56055fd9039df0f82c3950df71a7795647a639e551f12f6dbd81f67`.
- Static/LSTM prediction SHA-256: `fdc1189ffeec1993db78c4dbc133dea3d2f86dfcc2818d11395f64b220f4d00f` / `d2c60df2d162a48356377f6d20e2553bf55ff7b52a30794011fc1b317d8ee50d`.
- CPU-only; no GPU or cloud compute. No new model weights or data were created.
- Four focused regression tests pass; Ruff passes for the analysis source,
  runner and tests.
