# Replay teacher-target audit

The parent coverage trainer calls `choice_loss`, which averages hard-target CE
per question. The completed context replay also averages hard-target CE. Neither
uses saved teacher probabilities. This is existing behavior, not a new loss bug.

Read-only audit of the exact1,536 original training questions in replay order:

| Type | Count | Non-one-hot teacher | Mean gold probability | Mean entropy |
|---|---|---|---|---|
| binary/noul |633|561|.925285|.203913|
| choice |604|562|.741021|.728207|
| score |299|280|.650634|.874003|

All teacher argmaxes agree with hard targets (ties permitted). Score mapping is
verified against the full ordered criterion list; choice uses explicit IDs and
binary uses true/false IDs. Rounded source probabilities are retained as stored:
allow absolute sum error <=.001 for this descriptive audit; no normalization or
training change. Mean sum error is0/.00003113/.00003177 respectively. All1536 rows
map; no exclusions.1,403 have probability mass on more than one option.

This motivates a future one-factor training-only target comparison, not a causal
claim. Keep the completed run and labels immutable. A future soft-target arm must
explicitly normalize admitted rounded distributions, pin mapping and hashes, keep
context labels hard, and use the identical replay order, initialization, optimizer,
384-update budget and fixed final endpoint. Compare all existing metrics; do not
fit temperature on observed validation or tune teacher weights to that validation.
No soft-target learning has run yet. Teacher probabilities are synthetic annotation
outputs, not established calibrated truth; improved calibration is not guaranteed.

Reproduce: `python -m scripts.audit_replay_teacher_targets`. Adjacent JSON contains
source/order hashes, counts and rounding diagnostics. This audit reads training
teacher fields only; no validation teacher or sealed audit records are inspected.
