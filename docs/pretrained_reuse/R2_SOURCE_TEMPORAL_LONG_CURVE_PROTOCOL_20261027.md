# Source-candidate fixed 32-epoch learning curves

Repeat the existing joint baseline and visual-LSTM residual from fresh
initialization on the same three video-group folds and existing pinned
official-train cached features. Fix seed 17, AdamW LR 0.001, weight decay
0.0001, batch 32 and gradient norm clip 5. Both arms use identical per-fold
sampler permutations. Train **32 complete epochs** with no performance-based
early stopping, stopping only on nonfinite values or resource failure. The
previous 7-epoch models and all artifacts remain read-only.

Evaluate at fixed epochs 1, 2, 4, 7, 8, 16, 24 and 32. Save per-record OOF
predictions at those points, full online train-loss history and separate
checkpoint states at epochs 7, 16 and 32. No checkpoint or hyperparameter is
selected from these results. Original development/official validation/test
features or labels and sealed audit remain unopened. The 7-epoch prior result
informed this follow-up; these folds are observed development, not a blind test.

Compare both models at each fixed epoch with paired group-bootstrap
uncertainty; report accuracy, set NLL, multiclass Brier, ECE and visible/none
subgroups. Track the changing-weight training loss separately from validation
metrics; it is not a fixed-checkpoint generalization gap. This controlled
longer budget is a learning-curve diagnostic, not proof of grokking, a model
ceiling, causal temporal understanding or product quality. One seed and source,
overlapping training folds, oracle candidate tracks, shared participants and
unverified source labels remain limitations. Do not promote a candidate from
these results.
