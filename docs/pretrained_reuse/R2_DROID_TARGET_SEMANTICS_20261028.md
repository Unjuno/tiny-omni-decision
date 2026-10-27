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

### Instruction availability in the dense training cohort

The existing fresh-shard `audit.json` was joined by the existing canonical
source ID to the frozen 275-row `train.jsonl`; all IDs matched uniquely. The
audit aggregates normalized instruction strings over the entire episode and
over instruction fields. On that basis 150 recordings have at least one
instruction and 125 have none. By the original sparse target class, present /
absent counts are `stop 16/13`, `x- 5/3`, `x+ 59/43`, `y- 11/7`,
`y+ 10/13`, `z- 36/35`, `z+ 13/11`. These are recording counts, not dense
window counts. They describe this train cohort only and do not establish that
an instruction was available at a given window's prediction time. The audit
does not retain per-step availability; using its episode-wide text directly
could introduce future information. Sparse classes are also small.

The check read no media or validation data. Inputs:
`droid-fresh-next-action-prep-attempt-02/audit.json` SHA-256
`dea0dfa4a0eefd24afaad5278f2a23662626b7573ae03e7bd9a66d55397b27f8` and
`train.jsonl` SHA-256
`4e14850bbbe2eee2a96116a97ca935dfd5dea91d197c00639ea7ed02e2d93341`.
The join used `canonical_id` from `scripts/audit_droid_train_temporal.py`;
instruction strings and identifying paths were not copied into this report.
Any future conditioning experiment must audit step-level availability and
missingness before fitting and preserve instruction/group isolation.

### Past-index availability check

The pinned source shards were subsequently read for the 275 train recordings
only. For each dense target `t`, the three language fields were checked at
`t-1` and across `[0, t)`, without copying instruction text. Of 80,998 windows,
42,076 have nonempty text at `t-1`; the same number have text at any earlier
index. The source audit found text in 150 episodes. The remaining
38,922 windows have none in the stored past indices. This establishes serialized
index availability, not when an instruction was actually provided to an
operator or whether it identifies the correct future decision.

`scripts/audit_droid_instruction_timing.py` checks the pinned receipt, parser,
train manifest and each shard hash and verifies all 275 IDs and 80,998 windows.
The aggregate output is
`C:/CodexArtifacts/pretrained-reuse-r2/droid-instruction-timing-20261028-attempt-01.json`,
SHA-256 `51802f7c26c1e6df717f9d4a46a4ebcc82f7b211c627f021e28a17215b2611b0`.
Three CPU tests cover future exclusion, alternate instruction fields and length
validation; Ruff passed. No validation, sealed audit, model fitting or GPU was
used. A conditioning comparison remains subject to recording/instruction
isolation and explicit treatment of the missing half of these windows.

The existing episode-level instruction audit contains 334 distinct normalized
strings across these 275 recordings. Eight strings appear in more than one
recording, connecting 13 recordings in total. This count uses the entire
episode and therefore is a conservative grouping signal, not a safe model
input. A new grouped-CV comparison must form transitive components across
recordings sharing any normalized instruction before assigning folds; keeping
only recording IDs apart would not guarantee instruction isolation. The
existing train/validation split was already grouped by normalized instruction;
the new concern is partitioning within the training subset. The check used the
two pinned inputs listed above and printed aggregate counts only.
