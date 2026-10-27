# Baseline and visual-LSTM 32-epoch learning curves

The [fixed protocol](R2_SOURCE_TEMPORAL_LONG_CURVE_PROTOCOL_20261027.md)
completed all 32 epochs for both heads across three video-group folds on the
same 2,593 official-train records. This reuses already observed development
folds, with one seed and oracle candidates; it is not a blind or product test.
No new checkpoint or hyperparameter was selected. The original development,
official validation, test and sealed audit features/labels were not accessed.

| Epoch | Baseline Accuracy | Baseline NLL | Baseline Brier | Baseline ECE | LSTM Accuracy | LSTM NLL | LSTM Brier | LSTM ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4786 | 1.4562 | 0.5410 | 0.0781 | 0.4855 | 1.4531 | 0.5422 | 0.0902 |
| 2 | 0.5037 | 1.3809 | 0.5548 | 0.0520 | 0.5067 | 1.3798 | 0.5562 | 0.0572 |
| 4 | 0.5264 | 1.3267 | 0.5529 | 0.0323 | 0.5179 | 1.3496 | 0.5614 | 0.0368 |
| 7 | 0.5515 | 1.4441 | 0.5946 | 0.1240 | 0.5534 | 1.4505 | 0.5908 | 0.1238 |
| 8 | 0.5704 | 1.4632 | 0.5924 | 0.1209 | 0.5623 | 1.5192 | 0.5977 | 0.1324 |
| 16 | 0.5542 | 2.3005 | 0.6839 | 0.2511 | 0.5650 | 2.3184 | 0.6821 | 0.2382 |
| 24 | 0.5642 | 2.9740 | 0.7329 | 0.2788 | 0.5646 | 3.1771 | 0.7418 | 0.2826 |
| 32 | 0.5607 | 3.4879 | 0.7494 | 0.3103 | 0.5592 | 3.5317 | 0.7603 | 0.3116 |

The equal-fold mean of changing-weight online train NLL fell from 1.0851 to
0.0301 for baseline and 1.0776 to 0.0388 for LSTM between epochs 4 and 32,
while pooled held-out NLL, Brier and ECE worsened substantially. That is an
overfitting/calibration-degradation signal; online train loss is not a
fixed-checkpoint generalization-gap measurement. Accuracy rose from early
epochs but did not sustain a clear further gain after epoch 8. This run does
not establish a later grokking transition, nor rule one out under other
objectives, data, seeds, schedules or larger budgets.

At epoch 32, LSTM minus baseline Accuracy was −0.15 percentage points, with a
paired 320-video bootstrap interval of −1.56 to +1.26 points (10,000 draws,
seed 17). Earlier and later interval details, visibility strata and full curves
are in the summary. OOF models share training data between folds, sources may
share participants, and this one-seed interval does not measure training-seed
uncertainty. These results cannot support promotion or a four-modality 95%
claim.

CPU training completed in 190.08 seconds; CUDA was not initialized. All 18
saved checkpoints (epochs 7, 16, 32 for both arms and three folds) strictly
loaded. The six epoch-7 prediction hashes matched the earlier separate runs;
other checkpoint prediction parity was not independently replayed. Every
scheduled epoch's per-record predictions and all 192 online epoch-loss rows
are stored outside Git. No source artifacts were overwritten.

Training receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-temporal-long-curve-20261027-attempt-01/run.json`,
SHA-256 `619f9ea6c704477c8c2ddcb6ea3030e52aac470e5205dd550cb0ae0dba14d906`.
Summary:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-temporal-long-curve-summary-20261027-attempt-01/result.json`,
SHA-256 `b7f883f35693cacec799f2b4e27b98b369e3fd6a541d1eb57acd7f15e3098a8d`.
The run records the exact protocol, script, history, checkpoint and prediction
hashes. Four targeted tests and Ruff passed; the full CPU test suite passed
2,684 tests with 11 skipped. The preceding frame-order-control commit's CI
passed.
