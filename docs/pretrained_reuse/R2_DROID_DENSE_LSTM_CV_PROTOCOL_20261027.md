# Small LSTM on fixed dense DROID train windows

Freeze before training. Use all 80,998 pinned train windows, the exact five
prior recording-group folds, previous-command one-hot and eight past 14-value
sensor observations. No validation, audit, video, encoder inference or GPU is
used. Keep all natural-prevalence held windows. Compare against the pinned
change-weighted history-plus-sensor linear OOF predictions from attempt 02.

Train a unidirectional one-layer LSTM, input width 14, hidden width 8, no
dropout; concatenate its last past-window hidden state and the same seven-value
previous-command one-hot, then apply a seven-logit linear head. The LSTM/head
has 880 trainable parameters versus 840 for the comparison head. The LSTM uses
PyTorch's seeded default initialization (`17 + fold`); the earlier linear head
started at zero. This is a near-size comparison, not equal architecture or
optimization. Standardize each raw sensor coordinate with the outer training
fold's windows and time points only; use a 1e-6 scale floor and clip [-20,20].
The linear comparator standardized its invertible last-state/delta features;
different preprocessing and recurrent inductive bias are part of this test.

Use the same fold-specific stable/change training weights, batch size 2048,
AdamW 0.003, weight decay 0.01, epoch permutations seeded `17 + fold`, and 50
epochs as the earlier weighted run. All training windows are used; do not fit
normalizers or weights on held groups. Fail on nonfinite loss/gradient. Score
held windows once after epoch 50; no validation-driven checkpoint selection,
hyperparameter/seed tuning, or extra training run. Save every training loss,
five fold models, normalizers, weights and OOF probabilities outside Git.

Report raw weighted probabilities and the same fixed algebraic correction,
natural-prevalence Accuracy, NLL, macro NLL, Brier, ECE-10, per-recording mean
Accuracy, target-defined change exact accuracy, change-detection recall,
precision and stable false-change rate. Compare LSTM minus the fixed linear
baseline with recording-group paired bootstrap: 2,000 resamples, seed 17,
percentile 95% intervals for pooled Accuracy and NLL. No threshold or model
selection from these results. All results are post-hoc follow-up development
evidence on previously observed train groups, not blind or independent-site
validation. Repeated windows are correlated, axis-label changes can be noise,
and this sensor-only LSTM does not establish audio/video fusion or product
quality. A decreasing training loss alone does not establish a held plateau or
absence of overfitting. No final product model is created.
