# Fixed DROID weighted-head sensor interventions

The [fixed protocol](R2_DROID_WEIGHTED_SENSOR_INTERVENTION_PROTOCOL_20261027.md)
replayed the five saved history-plus-sensor heads on all 80,998 prior train-CV
windows. The unchanged-input replay matched the saved PyTorch OOF probabilities
within `7.75e-7`. It compared the aligned sensor, the outer-train feature mean,
and a non-overlapping half-recording cyclic sensor shift. Previous command and
target remained unchanged. The donor is from the same recording/fold but can
come from the future; this is a negative-control intervention, not a permissible
inference input. No model was trained or selected.

| Weighted-head input | Accuracy | NLL | Brier | ECE-10 | Change detection recall | Stable false-change rate | Exact changed commands |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aligned sensor | 0.84734 | 0.82747 | 0.38132 | 0.30934 | 14.53% | 7.30% | 444 / 7,442 |
| Outer-train mean sensor | 0.90812 | 0.95462 | 0.44117 | 0.46821 | 0% | 0% | 0 / 7,442 |
| Within-recording shifted sensor | 0.69155 | 1.16902 | 0.54949 | 0.24563 | 28.34% | 24.45% | 445 / 7,442 |

Detection means predicting any different command, not necessarily the correct
new command. Precision was 16.76% aligned and 10.50% shifted. The shift
increased detection and false alarms together; the number of exact new-command
predictions was almost unchanged. Mean-sensor Accuracy is inflated by the
dominant stable-command cases. Therefore neither raw detection recall nor
aggregate Accuracy alone establishes meaningful change prediction.

After the previously fixed weight correction, aligned / mean / shifted
Accuracy was 0.90696 / 0.90812 / 0.89602, NLL was 0.44286 / 0.45706 /
0.53838, and change recall was 0.44% / 0% / 1.92%. The correction was not
fitted and does not prove calibration.

Paired bootstrap resampled all 275 recording groups with replacement 2,000
times, seed 17, and used percentile 95% intervals. For weighted aligned minus
shifted input, pooled Accuracy difference was +0.15579 [0.14366, 0.16840] and
NLL difference was -0.34155 [-0.37445, -0.31290]. For weighted aligned minus
mean input, Accuracy difference was -0.06078 [-0.07420, -0.04842] and NLL
difference was -0.12715 [-0.14229, -0.11253]. Intervals describe this
previously observed train-CV cohort; folds share training data, windows within
recordings are correlated, and operator/site independence is not established.

The head is sensitive to the sensor intervention, but shifted sensors can
conflict with the unchanged action history and the shifted control produces
many more false alarms. This does not prove temporal reasoning, causality,
independent generalization, audio/video fusion, or the product quality goal.
No validation, audit or final-test data was accessed, and no threshold or model
was chosen using the intervention metrics.

Private artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-weighted-sensor-interventions-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`b103058ae5d2444f5ed333f107e01651383f9de75ef131ba17b8e92fb10dddeb`.
The summary includes all condition metrics, source/code hashes, bootstrap
intervals and the hash of the private OOF probability arrays. CPU scoring and
aggregation took 0.57 seconds after inputs were loaded.
