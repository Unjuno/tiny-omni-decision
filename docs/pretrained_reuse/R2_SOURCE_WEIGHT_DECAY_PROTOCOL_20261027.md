# Fixed stronger weight-decay comparison

The prior 32-epoch curves show falling online train NLL and worsening
held-out NLL/Brier/ECE. On the same three train-source video folds, train a
fresh joint baseline readout with **only AdamW weight decay changed from
0.0001 to 0.01**. Same frozen inputs, model, initialization seed 17,
permutation, LR 0.001, batch 32, gradient clipping 5, 32 epochs and fixed
evaluation epochs 1/2/4/7/8/16/24/32. No early stopping, checkpoint or
hyperparameter selection. Preserve the previous baseline and all artifacts.

Save per-record OOF predictions at each evaluation and separate checkpoints
at 7/16/32, and compare to the saved baseline at the same epochs with paired
video-group bootstrap. Stop on nonfinite values or if free disk falls below
1 GiB; do not delete existing data. Old development/official validation/test
and sealed audit are not opened. This is one deliberately chosen regularizer,
one seed and previously observed train-source folds: it is exploratory, not a
blind test, product selection, or proof of an optimal regularization schedule.
Record any negative result and the visible/none-visible tradeoff.
