# RoboMNIST TRAIN-only learning curve protocol — 2026-11-04

This scoped development experiment corrects the blanket no-further-fit conclusion
in the previous admission notes. The original objective and EVAL_PROTOCOL allow
small readout development over frozen features. The exposed original validation
remains closed to further selection/evaluation; no independent product test is
claimed. Fresh external data is needed for new product evidence, not for every
TRAIN-internal diagnostic. All previous results and restrictions on rights and
sealed audit remain in force.

## Question and fixed design

Does the existing rank-32 bilinear matching readout overfit before 400 updates?
The original fit attained train CE 0.00215, motivating this diagnostic. Before
adding temporal capacity, measure the same architecture's learning curve.

- Source: original pinned RoboMNIST TRAIN only, 780 groups / 390 complete pair
  units / 1,560 binary decisions. Original validation arrays are never loaded.
- Five seed-17 deterministic condition-stratified folds of intact 2x2 pairs;
  each underlying audio/video source occurs in exactly one fold. Existing
  scene/device limitations persist. These are internal development folds,
  not unseen data in the broader history of project development.
- Fresh initialization per fold; fit mean/std normalization on fitting groups
  only. No original checkpoint is loaded. Frozen encoders/features are reused.
- Original model and optimizer: rank32, AdamW lr0.01, weight_decay0.0001,
  norm clip5, full fitting batch, 400 updates, CPU four threads, seed17.
- Report steps 1/25/50/100/200/400 for every fold and concatenated out-of-fold
  predictions, with Accuracy/NLL/Brier/ECE-15 and train metrics. All scheduled
  checkpoints and exact reload parity are retained. No budget/checkpoint is
  selected for product promotion. Report all points, including worse results.
- Overfitting diagnosis: falling training NLL coupled with rising held-fold NLL;
  report calibration and accuracy trade-offs rather than selecting one metric.
- Folds share training data; pooled out-of-fold results do not constitute an
  independent final test. Any future selection from this curve is development.
- Output is a new external directory; at least 2 GiB free space required. No
  download, GPU/model loading, original checkpoint modification, or sealed audit.

Only after this curve is measured should a separate temporal comparison be
specified. Cached sequences retain audio[8,384] and video[8,768], but their clocks
are not aligned. A future recurrence must encode each stream separately, and
ordered/shuffled controls must share capacity. No LSTM is trained in this run.

## Follow-up fixed after the baseline curve, before the next fit

The baseline completed and showed rising held-fold NLL in every fold despite
falling training loss. Test one new arm: cosine score multiplier 1.0 instead of
sqrt(32), preserving every fold, fit-only normalizer, initialization, rank,
optimizer, update count, and reporting point. This keeps 73,728 parameters.
The new arm gets a separate output directory and never loads the old checkpoint.
This changes optimization as well as output confidence; it is not post-hoc
calibration. Unit scale caps confidence at sigmoid(2), about88%, so it is a
mechanism diagnostic rather than a product confidence policy. A reduction in
NLL without improved discrimination is not a quality-gate pass. No temporal
capacity is added. All scheduled points remain primary descriptive outputs;
no seed/budget selection or final validation follows this experiment.
