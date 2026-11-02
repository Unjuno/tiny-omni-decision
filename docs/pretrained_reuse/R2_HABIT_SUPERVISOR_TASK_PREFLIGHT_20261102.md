# HABIT Supervisor transition-choice preflight — 2026-11-02

## Status

The initial metadata source screen remains `REVIEW_NOT_ADMITTED` for quality
claims. This follow-up freezes a **low-power exploratory** RGB-video plus robot
state decision protocol; it does not establish an actor-independent benchmark,
close R2-M, or justify product promotion. No model has been fitted on this
protocol yet. Sealed audit data are not part of this source.

## Source and frozen split

- Dataset: `configinc/HABIT`, Hub revision
  `bae97d251907743f8fb6db7e70c61d361df4dc1e`; license declaration and
  participant protections are recorded in
  [the source screen](R2_HABIT_SOURCE_SCREEN_20261102.md).
- Split seed: 17. Entire `task_details` filename groups, including joined SID
  filenames, are indivisible. The plan has 15 train groups / 360 episodes, 3
  validation groups / 48 episodes, and 3 reserved groups. No reserved media or
  labels are included in the transfer list.
- Metadata-plan SHA-256:
  `fe9d8012131db25578d25a6501a9c56c50dd2d4ddef33cfd2145d3aa81cb8f1b`.
- Ordered train episode IDs SHA-256:
  `8d518e20ae8702237b9cc764982c13c054ea808a75a54aecc6c367ce8334d477`
  (360 IDs). Ordered validation episode IDs SHA-256:
  `305a037f422b549292de08e57ec2f2699d5ccab1460984db336f19911ff61193`
  (48 IDs). The split audit found zero train/validation episode, task-group,
  exact video SHA, or Parquet SHA overlap.
- This is only 3 validation task groups and there are no participant/actor IDs.
  Results cannot support actor-independent generalization or the roadmap's
  quality-admission gate, regardless of the number of frames or episodes.

## Fixed decision task

Use Supervisor-role Lint Rolling episodes with a supplied set of that task's
robot low-level action descriptions. At an annotated change point `b` in
`low_level_task_index`, use eight synchronized `human_front_view` frames and
robot-observation state rows at indices `b-12..b-5`; the label is the action
ID at `b+5`. The dataset is 10 Hz, so inputs end 0.5 seconds before the
transition and the target is 0.5 seconds after it. This is a transition-focused
lookahead task, not an always-on policy. Drop boundaries without the full fixed
history/horizon. Fail closed if the future label is not among the supplied
options.

Candidate IDs are looked up by exact text in the pinned global
`full/meta/subtasks.jsonl` table and checked against each episode's
`low_level_tasks`. `task_index` addresses the high-level task table;
`low_level_task_index` addresses the distinct robot-subtask table. For the
train-only smoke episode 1062, high-level ID 7 maps to “Clean the surface of the
table,” while low-level IDs 15–18 map to its four Lint Rolling actions. This is
not an index mismatch; the two ID spaces must not be conflated.

Inputs must not include `sid`, task/subtask IDs, high-level or low-level task
text, future state/action values, annotation flags, or `human_role_subtask_index`.
The action descriptions are supplied candidate options, not labels embedded in
the observation. Model selection uses only the three held-out validation task
groups; reserved groups remain unopened.

## Planned controls and reporting

With identical sample IDs, target/options, frozen encoders, seed, and readout
budget, compare options-only, video+options, state+options, and video+state+
options. Include missing/shuffled-video and missing/shuffled-state controls.
Report Accuracy, NLL, Brier, ECE-15, option-only baseline, per-task-group
results, confidence intervals clustered by task group, train/validation gap,
sample counts, parameter counts, extraction/training/evaluation time, and peak
VRAM. Any estimate is exploratory because there are only three validation
groups. A high episode-level score cannot override this limitation.

Use frozen pinned VideoPrism and the already audited frozen MiniLM option
encoder; train only a compact decision readout. Compare mean-pooled temporal
features with a small recurrent temporal readout only if the common baseline
pipeline is valid. Do not unfreeze either encoder or use reserved groups for
model/architecture selection.

## Acquisition/runtime smoke

One selected train-only episode (1062) was downloaded from the pinned revision;
its video and Parquet match the publisher size and SHA-256. PyAV decoded the
10-fps H.264 video as 457 frames at 640x480; its 457 Parquet rows align in
count. The repo audit script also resolved its four global option IDs, checked
three transition boundaries, and emitted three eligible windows with no edge
skips. The script SHA-256 is
`331c20fb24e749cdd10623f718e161785e66611b86168b91a793465b6dd82655`; the
external smoke report at
`C:/CodexArtifacts/pretrained-reuse-r2/habit-task-feasibility-20261102-attempt-01/transition-audit-train-smoke.json`
has SHA-256
`b28f4f4c8d7b38c2c674f3535f93230f9eff28028576123318f99b40a5edaf6a`. The
source's `info.json` codec declaration is not treated as runtime truth; codec
is detected from the media stream. The smoke confirms decoding and annotation
availability only, not model quality.

Train/validation media acquisition started in the external attempt directory
named in the live run receipt. The downloader verifies every file's size and
SHA-256 and refuses reserved entries; its progress and completion receipt are
external artifacts, not model results. Before extraction or fitting, run the
data-integrity and event-count checks over the frozen selected split, then
report the three-group validation limitation alongside any exploratory
metrics.
