# Fixed weight-decay comparison

Following the [fixed protocol](R2_SOURCE_WEIGHT_DECAY_PROTOCOL_20261027.md),
we trained the same 344,834-parameter audio/video readout for 32 epochs in
three video-group folds, with AdamW weight decay 0.01 instead of 0.0001.
The same frozen features, seed, fold assignments, batches, optimizer learning
rate, and evaluation epochs were used. No model or checkpoint was selected.
The 2,593 examples are previously observed official-train development data,
with oracle video candidates; this is not a blind or product evaluation.

| Epoch | Original accuracy | Original NLL | Original Brier | Original ECE | 0.01 accuracy | 0.01 NLL | 0.01 Brier | 0.01 ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.5264 | 1.3267 | 0.5529 | 0.0323 | 0.5249 | 1.3224 | 0.5503 | 0.0319 |
| 7 | 0.5515 | 1.4441 | 0.5946 | 0.1240 | 0.5550 | 1.4551 | 0.5943 | 0.1273 |
| 8 | 0.5704 | 1.4632 | 0.5924 | 0.1209 | 0.5658 | 1.4337 | 0.5830 | 0.1155 |
| 16 | 0.5542 | 2.3005 | 0.6839 | 0.2511 | 0.5530 | 2.2285 | 0.6805 | 0.2357 |
| 24 | 0.5642 | 2.9740 | 0.7329 | 0.2788 | 0.5692 | 2.8883 | 0.7280 | 0.2768 |
| 32 | 0.5607 | 3.4879 | 0.7494 | 0.3103 | 0.5704 | 3.3645 | 0.7441 | 0.3042 |

At epoch 32, the paired accuracy difference was +0.96 percentage points;
the 95% video-group bootstrap interval was -0.31 to +2.26 points (320
videos, 10,000 draws, seed 17). NLL improved by 0.123, but remained much
worse than epoch 4. Visible/none-visible accuracy was 0.5187/0.6763 for
baseline and 0.5245/0.6965 for 0.01 at epoch 32. Stronger decay alone did
not resolve the observed long-run calibration degradation. One seed and
reused development folds do not establish a general regularization effect,
source independence, or an ultimate performance ceiling.

The CPU run took 67.99 seconds without initializing CUDA. Nine checkpoints
were hash-checked and strictly loaded; per-example predictions were checked
against receipts and paired by ID and targets. Prediction replay from the
checkpoints was not performed. Original data and checkpoints were preserved.
The exact executed script is archived alongside the run; the repository copy
has only import-order and unused-import cleanup after execution.

Run receipt: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-weight-decay-20261027-attempt-01/run.json`, SHA-256
`ae9b3ad3f2ecf6240c13659857fd1968e495da5ee438f556374a2f1c65ce6e51`.
Summary: `C:/CodexArtifacts/pretrained-reuse-r2/perception-source-weight-decay-summary-20261027-attempt-01/result.json`, SHA-256
`e6e3ab7782469ae01d03364bcd4303435ad937cdbc58a0d13281d2c7847ab3d9`.
No VisTouch or UnAV-100 data was used; no sealed audit was opened.
