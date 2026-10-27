# DROID train-only instruction probe: result

The [fixed protocol](R2_DROID_INSTRUCTION_PROBE_PROTOCOL_20261028.md) ran on
275 existing train recordings in five instruction-connected folds, using
only text at the index immediately before the target. There were 268 connected
components and 150 rows with nonempty text. All 275 received out-of-fold
predictions; no development-validation or sealed-audit result was scored.

| Inputs | Rows | Accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Sensor | 275 | 36.36% | 2.6591 | 0.9121 | 0.3063 |
| Sensor + hashed text | 275 | 37.09% | 2.7879 | 0.9179 | 0.3196 |
| Sensor, text present | 150 | 37.33% | 2.7794 | 0.9091 | 0.3081 |
| Sensor + hashed text, text present | 150 | 36.67% | 2.9701 | 0.9237 | 0.3559 |

Two more correct rows overall do not outweigh the worse NLL, Brier and ECE.
Within text-present rows accuracy and all probability metrics also worsened.
The 200-update fixed linear readouts are poorly calibrated. This does not rule
out informative instructions, pretrained text encoders, video or other training
strategies; it is a small, nonblind, exploratory comparison on one narrow
robot command task. The text and sensor-plus-text arms have different numbers
of parameters and no causal intervention was performed. No model is promoted.

Class counts in option order are `[29, 8, 102, 18, 23, 71, 24]`. See the
result for fold-level metrics and pinned inputs. Output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-instruction-probe-20261028-attempt-01/`,
`result.json` SHA-256
`86f5090879e965d26c83447b51390709a389295afbb7220de5e778f752b00320`.
`oof.npz` contains probabilities, labels, folds and text-present flags; no
instruction text is saved. Three focused CPU tests and Ruff passed. Both arms
ran on CPU; no existing checkpoints or feature caches were modified.
