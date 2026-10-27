# DROID instruction probe learning curve

The [fixed replay](R2_DROID_INSTRUCTION_LEARNING_CURVE_PROTOCOL_20261028.md)
ran five train-only folds and checked the 200-update predictions were exactly
equal to the previous saved probabilities. Mean training CE is an unweighted
mean of the five fold means; held metrics pool all 275 out-of-fold rows.

| Update | Input | Mean train CE | Held accuracy | Held NLL | Held Brier | Held ECE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Sensor | 1.888 | 15.64% | 1.977 | 0.863 | 0.102 |
| 1 | Sensor + text | 1.677 | 29.09% | 1.806 | 0.808 | 0.040 |
| 10 | Sensor | 1.402 | 28.36% | 2.023 | 0.843 | 0.115 |
| 10 | Sensor + text | 1.193 | 33.45% | 1.973 | 0.827 | 0.155 |
| 50 | Sensor | 0.848 | 34.91% | 2.079 | 0.849 | 0.203 |
| 50 | Sensor + text | 0.658 | 34.91% | 2.241 | 0.855 | 0.237 |
| 100 | Sensor | 0.625 | 35.27% | 2.289 | 0.878 | 0.254 |
| 100 | Sensor + text | 0.478 | 36.73% | 2.537 | 0.890 | 0.282 |
| 200 | Sensor | 0.454 | 36.36% | 2.659 | 0.912 | 0.306 |
| 200 | Sensor + text | 0.349 | 34.18% | 2.959 | 0.935 | 0.363 |

Training CE falls while held NLL rises throughout these observations for both
heads, a clear overfitting/calibration warning on these train-derived folds.
Early text-plus-sensor accuracy and NLL are better than the same-step sensor
arm, whereas at 200 updates they are worse. Since these curves were examined
after prior results, they do not justify selecting an early checkpoint or
claiming independent improvement. They do show that the 200-update outcome
alone cannot identify whether the inputs carry useful information. Additional
group-isolated data and a prospectively frozen selection protocol are needed.

The external artifact is
`C:/CodexArtifacts/pretrained-reuse-r2/droid-pretrained-instruction-curve-20261028-attempt-01/`.
`result.json` SHA-256
`ae4e3dc4e279e3d9e166ce6d11eb9b30f6ed9fd457792ebe02f25fb2239188b9`.
All milestone OOF probabilities and fold-level metrics are retained; no
instruction text is saved. No held-out validation, sealed audit, GPU, or prior
checkpoint was modified or used for model selection.
