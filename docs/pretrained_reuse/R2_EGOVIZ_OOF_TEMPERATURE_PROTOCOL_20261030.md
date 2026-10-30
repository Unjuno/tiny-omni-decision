# Training-session cross-fitted temperature — protocol

After the observed in-sample scalar overconfidently worsened development NLL,
test a scalar estimated from predictions on training sessions omitted from the
models that make those predictions. This prospective follow-up is exploratory:
the original checkpoint and previous calibration comparison already used the
development set. It is not a blind audit or an independent product claim.

Keep the original joint scorer, original checkpoint, data, candidates and
development IDs unchanged. Partition its 24 training session groups into five
folds using sorted IDs and NumPy default RNG seed 17. For each fold, omit those
groups from fitting and choose the next fold cyclically as an inner validation
set; use only the remaining three folds to fit the same 81,920-parameter
scorer. Fix seed 17 separately for each fold, AdamW 0.003, weight decay 0.01,
batch 32, max 100 epochs and NLL patience 12 / min_delta 1e-6. The inner
validation never includes the corresponding held-out group. Save all five
selected fold weights and their metadata. Each training example gets exactly
one prediction from a model that did not fit or select on its session.

Fit a single positive inverse temperature on all 332 such training-only
out-of-fold logits using the existing deterministic convex scalar optimizer.
Only after fitting this scalar, apply it to the original fixed checkpoint's
307 development examples and compare with the unchanged original probabilities
and previously observed in-sample-trained scalar. Positive scaling must not
change option rankings. Record Accuracy, NLL, Brier, ECE, mean confidence,
paired session-bootstrap intervals, all sample predictions, hashes and the
five fold histories. Run once on CPU with separate output; keep sealed data
unopened and protect all prior artifacts.

The five cross-fitted models train on fewer sessions than the original model;
their score distribution may not match it. Result interpretation must state
this limitation and the adaptive motivation following the earlier diagnostic.
Regardless of outcome, no model or scalar is promoted on these development
observations; an independent calibration source and held-out audit remain open.
