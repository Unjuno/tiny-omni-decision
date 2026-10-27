# DROID dense-window target semantics

This is an inspection of the existing train-only construction, not a new model
result. For target index `t`, `make_windows` reads observation indices `t-8`
through `t-1`, previous command `t-1`, and target command `t`. The sensor
features are joint position, Cartesian position and gripper position. The
target is the dominant signed XYZ component of `action_dict/cartesian_velocity`,
discretized to seven classes. Language instructions and images are not inputs.
The constructor and frozen protocol are
`scripts/build_droid_dense_train_windows.py` and
`R2_DROID_DENSE_TRAIN_WINDOWS_PROTOCOL_20261027.md`.

The upstream DROID [dataset schema at commit
33ae6a67274f36d2e29525b86f23a56616ef43a7](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/docs/the-droid-dataset.md)
describes this field as commanded Cartesian velocity and exposes language
instructions and camera observations separately. Its [repository at the same
commit](https://github.com/droid-dataset/droid/tree/33ae6a67274f36d2e29525b86f23a56616ef43a7)
describes collection of teleoperated demonstrations. The current experiment
therefore predicts a future operator command class from prior proprioception and
the previous command; it is not a direct test of a condition-aware multimodal
decision. Missing intent and scene information may contribute to uncertainty,
but their contribution has not been measured.

Index exclusion establishes no target-row data is explicitly read into the
input. It does not establish physical synchronization, the acquisition latency,
or that all channels were available in real time in precisely this order.
Per-step physical timestamps were not retained in the constructed arrays.
Consequently the results cannot establish a physical reaction-time guarantee or
separate uncertain future operator choices from representation limitations.

The 200-epoch curve and persistent-change analysis remain development evidence:
see `R2_DROID_CHANGE_PERSISTENCE_20261028.md`. Longer fitting improves the
number of correctly predicted changes while worsening other measures; it does
not show a clean general capability improvement. A useful next comparison must
fix the task and inputs before evaluation, retain group separation, and measure
whether task instructions and scene evidence are actually informative. Any
new use of images, timestamps or language requires a fresh source, rights and
alignment audit; adding them cannot retroactively establish the performance of
the existing sensor-only model.

## Existing sparse video comparison

The separate `codex/r2-droid-fresh-holdout-20261026` worktree already evaluated
video and sensor inputs on a sparse next-action task using 275 train and 150
development-validation recordings. That earlier task uses eight uniformly
spaced points in the first quarter of an episode, unlike the consecutive dense
windows discussed above, and does not supply the previous command. Frozen
VideoPrism features are cached for those sparse examples only. With the
train-selected stronger L2 setting, validation accuracy was 38.67% video,
40.67% sensor, and 40.00% video plus sensor; the paired interval did not
establish a reliable fusion gain. The validation was inspected more than once.
Do not compare these numbers directly to the dense-window LSTM curve or repeat
that already executed comparison. See the other worktree's
`docs/pretrained_reuse/R2_DROID_FRESH_NEXT_ACTION_RESULTS_20261026.md` at
commit `eba818067c6a9ef9089288205c26f6a8254ca6e5` for its complete results.

That worktree's `R2_DROID_SELECTED_SHARDS_PROTOCOL_20261026.md` reports task
instructions present for 302 of 539 episodes in a different selected-shard
cohort. This does not establish instruction availability, usefulness or
balanced coverage in the dense cohort. Those properties must be measured from
the exact selected records before proposing an instruction-conditioned
experiment; missing instructions must not be silently substituted.
