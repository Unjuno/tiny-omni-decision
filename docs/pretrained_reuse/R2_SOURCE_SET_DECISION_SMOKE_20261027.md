# Source set-decision prototype: CPU smoke

The independent implementation follows
[the fixed plan](R2_SOURCE_SET_DECISION_PLAN_20261027.md). It keeps the
344,834-parameter candidate model unchanged and replaces neither existing
training code nor checkpoints. The output is a none-visible sigmoid and one
conditional sigmoid per valid candidate. The conditional sigmoid scores are
not a normalized distribution over nonempty sets and are not demonstrated to
be calibrated. The threshold rule is a provisional point decision.

A separate CPU process verified all pinned train input hashes, loaded existing
frozen features, selected one training example each with zero, one and two
visible target candidates, then ran the existing candidate model with the new
loss, backward pass and decoder. Loss was 1.1427212; all computed gradients
were finite; outputs were nonempty and none-visible was exclusive. This is a
wiring smoke on a randomly initialized model, not training or evidence of
accuracy or calibration. CUDA was not initialized; no model or data was saved.

The new targeted tests cover multi-target gradients, padding invariance,
none-visible exclusivity, fallback, invalid labels and invalid masks. Four
prototype tests and three set-audit tests passed, as did Ruff. A measured
comparison remains necessary under a separate frozen protocol; source grouping,
independent validation, multiple-modality utility and storage remain open.
