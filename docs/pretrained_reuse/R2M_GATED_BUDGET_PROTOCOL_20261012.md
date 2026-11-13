# Separate gated-readout learning-budget experiment

The100-epoch run is complete and immutable. At its endpoint both train and
validation NLL were still falling (1.6063/1.7291), so that result cannot establish
an optimization ceiling. Test a separate, fixed1,000-epoch run to distinguish
budget sensitivity from a claim that this architecture cannot learn. This is
not a silent extension or optimizer resume of the old artifact.

Freeze before execution: same seed17, fresh initialization, width64,421826
parameters, same frozen feature hashes and130train/39validation trials,
normalization/scaling, placement context,13 one-hot options, full-batch AdamW
lr0.001/weight_decay0.01, CPU float32/four threads. No scheduler, augmentation,
new sources, encoder updates or calibration. Same minimum-validation-NLL rule
including epoch0; keep best and final separately. Preserve every epoch's train/
validation probabilities, metrics and gradient norm. One run, no seed search.

Only changes to the runner: description, distinct output directory and epoch
budget100->1000. A source-equivalence test checks this restriction. Verify all
epoch0..100 probability/target arrays against the prior run, not just endpoint
Accuracy. If they differ, report the comparison limitation; do not conceal it.

Primary budget comparison: fixed100 vs1000. Report250/500/750 and selected
checkpoint too, without substituting selected performance for fixed endpoint.
Check the same predefined input shuffles and missingness interventions for the
selected checkpoint. Compare with sensor, video and scaled-linear baselines,
using aligned IDs and paired subject-cluster NLL intervals. No adoption merely
for improving over an undertrained100-epoch readout.

This reused three-subject validation supports a developmental learning curve,
not independent generalization or all-modality95%. A falling train loss with
rising validation NLL is evidence of overfitting. Finish the fixed budget unless
numerical/data/resource safety fails; do not start additional budgets afterward
without a distinct justification and protocol. No sealed audit or paid compute.
