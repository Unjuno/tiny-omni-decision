# DROID instruction probe learning curve

Repeat the previous pretrained-instruction experiment with identical inputs,
folds, initialization, optimizer and 200 updates, recording full-batch training
loss and held-fold probabilities at updates 1, 10, 50, 100 and 200 for both
arms. Compare pooled out-of-fold Accuracy, NLL, Brier and ECE at every fixed
point. Verify the 200-update predictions against the previously saved run.
This is retrospective analysis of an already observed training cohort and is
not used to select a product model or inspect a validation or audit split.
Different steps use the same dependent folds and are not independent trials.
