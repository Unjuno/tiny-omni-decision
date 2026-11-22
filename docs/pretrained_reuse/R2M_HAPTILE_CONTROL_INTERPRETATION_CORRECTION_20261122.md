# HapTile control interpretation correction — 2026-11-22

## Reason for correction

A code-to-report audit found that the description of the existing validation
interventions was stronger than the implementation supports. The original
checkpoints, prediction files, metric values, training history, split, and
feature caches were not changed or replayed. This note corrects interpretation
only.

## What the implementation did

- `missing_video` replaced the values in the complete four-stream video tensor
  (front camera, wrist camera, left tactile camera, right tactile camera) with
  zeros. The full-mode scorer still received all-true video validity and
  presence masks. This is a zero-value replacement, not a missing-video input.
- `missing_sensor` replaced all 25 state channels with their train-split means.
  Those channels are pose (3), end-effector velocity (6), joints (6), joint
  velocity (6), gripper position (1), and three scalar tactile fields. The
  separate left/right tactile-camera streams were not removed. The full-mode
  scorer still received all-true sensor validity and presence masks. This is a
  train-mean state replacement, not a missing-sensor input.
- `task_group_shuffled_video` and `task_group_shuffled_sensor` use
  `_shuffle_values_across_task_groups`: sorted task groups are mapped to the
  next group, and source rows cycle in order. These are cross-task replacements,
  not within-task shuffles. When group sizes differ, donor rows can be reused,
  so the mapping is not a bijective permutation. Only values are replaced;
  recipient timestamps remain unchanged.
- The task cutoff is the midpoint of the complete episode pose-log time range.
  Inputs are causal conditional on that cutoff, but the experiment does not
  test an online trigger that can choose the cutoff before the episode ends.

## Corrected reading of the saved metrics

The original NLL values are preserved: unmodified full model 1.120;
zero-value replacement of all four video streams 1.650; train-mean replacement
of all state channels 2.662; cross-task video replacement 2.170; and cross-task
state replacement 2.896. They show sensitivity to these specific value
perturbations on the selection-affected five-task validation set. They do not
establish modality absence/necessity, within-task modality dependence, robust
cross-modal reasoning, or real-time decision triggering. In particular, the
previous wording “shuffling video within task groups” is superseded by
“cross-task video value replacement.”

No validation predictions were rescored, no checkpoint was loaded, no model
was trained, and no data or sealed audit was accessed for this correction.

## Reproduction evidence

- Repository head before this correction: `13240541e382809da8e1e894aaa7fea5399507ac`.
- The saved executed training script is byte-identical to the tracked script:
  SHA-256 `290c7a175e5ed2d8fe97399b46d83f39c0fc82e9874efc1ba0cc4e9dee0d4e04`.
- Saved `validation-stream-controls.json` SHA-256:
  `4a6d106ef29a6bbb9702010d7f900502cef9d73e45845ca62242f02545a184c0`.
- Saved `experiment-report.json` SHA-256:
  `75f88f59550417db525b1372ea02d39e2f7e4789ec66e56952ace4028a1fd29e`.
- The external run remains under
  `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\fit-seed17-v1-attempt-01`.
- Synthetic CPU regressions pin donor cycling for unequal task sizes, state
  replacement with unchanged video values, and full-mode presence masks.
- Verification: focused tests `8 passed`; full CPU suite `3,273 passed,
  11 skipped`; `ruff check .` passed; manifest input/result hashes match.

See the unchanged result tables with corrected control wording in
[`R2M_HAPTILE_TEMPORAL_DECISION_RESULTS_20261122.md`](R2M_HAPTILE_TEMPORAL_DECISION_RESULTS_20261122.md)
and the cutoff note in
[`R2M_HAPTILE_ACTION_CHOICE_PROTOCOL_20261122.md`](R2M_HAPTILE_ACTION_CHOICE_PROTOCOL_20261122.md).
