# Dense DROID windows: fixed grouped-CV sensor ablation

Freeze before fitting. Use the new 80,998 train windows and five previously
fixed recording-group folds, verifying every dataset artifact hash and group
partition. No validation, audit, video, encoder inference or GPU is used.
Every held fold contains all its natural-prevalence windows; no change-only
filtering, oversampling, loss reweighting or dataset-size reduction occurs.
Input uses only the previous command and eight past sensor observations.

Compare a 7-input previous-command one-hot linear head with a 119-input head
using the same one-hot plus final 14-value sensor state and seven successive
14-value sensor differences. Output seven logits; both heads include bias and
have 56 and 840 trainable parameters. For each fold standardize sensor columns
with that fold's training rows only (scale floor 1e-6, clip [-20,20]); history
one-hot is not standardized. Train from zeros with fixed AdamW learning rate
0.003, weight decay 0.01, minibatch 2048, 50 epochs, and seed 17 plus fold ID.
Both heads use the same deterministic epoch permutations and all training rows.
Stop on nonfinite loss or gradients. Record training loss each epoch and score
held windows once after the fixed 50th epoch; never use held metrics to change
settings or select checkpoints.

Also compute the add-one seven-by-seven transition-table baseline from each
outer training fold. Report full out-of-fold Accuracy, macro Accuracy, NLL,
macro NLL, Brier, ECE-10, per-class metrics, per-recording mean Accuracy, and
target-defined stable/change counts and metrics. The primary comparison is
history-plus-sensor minus history-only macro NLL; no threshold, seed or epoch
selection follows observed held results. Save all outer predictions, heads,
training curves, normalizers, folds and hashes outside Git.

This is development cross-validation on previously observed train recordings,
not blind or independent-site validation. Windows within a recording are
correlated. Changes can reflect dominant-axis fluctuations; class and natural
prevalence remain unchanged. The ablation compares unequal parameter counts
and does not establish a general sensor, temporal-model, audio/video, fusion,
or product-quality conclusion. No final product model is created.
