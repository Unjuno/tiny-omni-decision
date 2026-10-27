# DROID LSTM fixed 200-epoch train-fold learning curve

The [predeclared protocol](R2_DROID_LSTM_LONG_CURVE_PROTOCOL_20261027.md)
completed five recording-group folds on 80,998 previously used train windows.
The 880-parameter sensor/history LSTM was freshly initialized for each fold;
all five epoch-50 raw predictions passed the prior-run replay tolerance. This
is development data, not independent validation, and neighboring windows are
correlated. No product checkpoint or sealed audit was used.

| Epoch | Raw accuracy | Raw NLL | Raw Brier | Raw ECE | Change recall | Exact changed command | Stable false-change rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.87574 | 0.87748 | 0.40256 | 0.38018 | 6.07% | 156 / 7,442 | 3.78% |
| 100 | 0.85440 | 0.88731 | 0.41061 | 0.35658 | 9.38% | 230 / 7,442 | 6.23% |
| 150 | 0.84049 | 0.89733 | 0.41594 | 0.34234 | 11.01% | 269 / 7,442 | 7.81% |
| 200 | 0.83547 | 0.90208 | 0.41877 | 0.33639 | 11.68% | 303 / 7,442 | 8.41% |

Longer training increases detected changes and correct changed commands, while
also increasing false changes and worsening overall accuracy, NLL and Brier.
ECE improves numerically but remains large. With the fixed weight correction,
accuracy changes from 0.90812 to 0.90801 and NLL worsens from 0.45328 to
0.46169 between epochs 50 and 200; exact changed commands remain 0 / 7,442.
Thus neither output is an acceptable demonstrated operating point. Training
loss generally falls while held-fold NLL worsens, consistent with overfitting
under this objective; weighting and changing prediction behavior also affect
interpretation. This does not establish a general LSTM or temporal ceiling.

For epoch 200 minus 50, the paired recording-group bootstrap (275 groups,
2,000 seed-17 resamples, percentile 95% interval) gives raw accuracy -0.04027
[-0.04793, -0.03332] and NLL +0.02460 [+0.01786, +0.03196]. The cohort has
already informed development, and these intervals are not a blind test.

CPU execution took 474.73 seconds with PyTorch 2.6.0+cu124 and four threads.
Milestone weights and held predictions are saved separately, without optimizer
or random states; they are not exact-resume checkpoints. Output directory:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-lstm-long-curve-20261028-attempt-01/`.
SHA-256 `summary.json`:
`0b973f95809c81572ef7a0496e955b43ea730fcadc2ffe06f5e98565ecdede16`;
`oof-probabilities.npz`:
`11d0c40026dc1b54d1bec6266371e640ede88a3d020709e37ecd7fa5fa962415`.
The summary records individual milestones, training curves, source hashes and
all artifact hashes. No independent multimodal or 95–98% claim follows.

An independent aggregation check reopened all 20 saved held-fold prediction
files and the combined OOF file. Each row appeared exactly once at each epoch;
the maximum absolute difference was `2.99e-8` across raw and corrected
probabilities. The small corrected-probability difference reflects the OOF
array's float32 storage. This checks aggregation, not independent model replay.
