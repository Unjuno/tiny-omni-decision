# RoboAug UR5e session-group metadata screen — 2026-11-13

## Disposition

**Do not acquire more RoboAug episodes or start a quality run for this candidate.**
The official tree contains 789 UR5e episodes across four observable date
prefixes, and each task is confined to one date. The source does not establish
that episodes within a date are independent collection groups. Even treating
each date prefix as one group, the effective count is four, far below the
repository's provisional 200-independent-group quality gate. The date/task
confounding also prevents a clean within-task held-out-date evaluation.

RoboAug remains a possible small engineering pilot, but it cannot support the
project's high-confidence task-quality claim on this evidence. The three
already downloaded sample episodes remain unchanged. No additional media,
features, models, or evaluation data were accessed; no GPU work was run.

## Pinned metadata audit

- Dataset: `X-Humanoid/RoboAug-Datasets`
- Revision: `08622781145a3791e8f7f26978ca162689da0c66`
- Scope: metadata-only listing of `trajectory.hdf5` paths, sizes, and object
  identifiers under all five `Single_Arm_UR_5e` success-task directories.
- Total: 789 episodes, 149,416,412,440 bytes (about 149.4 GB decimal).
- Full API listing receipt (stored outside Git):
  `D:\CodexArtifacts\pretrained-reuse-r2\roboaug-session-group-screen-20261113-attempt-01\episode-metadata-inventory.json`
- Receipt SHA-256:
  `3536df2cb5eaf09b11610f8b35217c4d3dbfea1828594812c0dfc51f51459ffc`

| ID date prefix | Episodes | Tasks |
|---|---:|---|
| `0307` | 294 | `ur_put_corn_into_the_pot` |
| `0313` | 295 | `ur_move_lemon_from_plate_to_bowl` |
| `0523` | 150 | `ur_stack_bowl_250523`, `ur_put_carrot_and_close_drawer` |
| `0524` | 50 | `ur_open_drawer_and_put_corn` |

Each of the five tasks occurs on only one date prefix. The date prefix is an
observable grouping proxy, not proof of independent sessions or participants.
The official README says no train/validation split is supplied. A random
episode split would therefore allow shared collection context across folds;
holding out dates would entangle task identity with date and leave at most one
date-level group for final evaluation.

The pinned official README defines `is_intervene=1` as a human-intervention
frame. The sample's all-zero values are not a task label and will not be used
as a label or filtering rule. The executed `puppet/*` signals remain the
appropriate behavior targets if a separate pilot is ever justified.

## Decision and next step

The source is not admitted for the primary Path A quality evaluation. Do not
spend roughly 149 GB on this task family for a benchmark that cannot pass the
existing sample/group floor. Continue R2 candidate admission for a source with
at least 1,000 usable decisions and 200 defensible independent groups; freeze
the split before any feature extraction. If a future systems-only RoboAug
pilot is proposed, label it explicitly as low-power engineering evidence and
do not treat it as a quality or multimodal-fusion result.

## Sources

- [Pinned official RoboAug robot-dataset README](https://github.com/Open-X-Humanoid/RoboAug/blob/3bd6ac54d9b0c6ad5a35ad1298f8f09545495631/dataset/robot_dataset/README.md)
- [Pinned official RoboAug dataset README and license](https://github.com/Open-X-Humanoid/RoboAug/blob/3bd6ac54d9b0c6ad5a35ad1298f8f09545495631/dataset/README.md)
- [R0 provisional quantitative acceptance policy](R0_QUANTITATIVE_ACCEPTANCE_POLICY_20261031.md)
- [Initial RoboAug source/task screen](R2_ROBOAUG_VIDEO_PROPRIO_SOURCE_SCREEN_20261113.md)
