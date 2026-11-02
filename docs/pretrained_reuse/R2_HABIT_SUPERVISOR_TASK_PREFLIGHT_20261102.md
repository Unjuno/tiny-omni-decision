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
skips. The executed script SHA-256 is
`688bc3b556094a379c16de4ce8faeffa7fa96d317af81081ee79922cdfa54fab`; it
refuses to overwrite an existing report. The external smoke report at
`C:/CodexArtifacts/pretrained-reuse-r2/habit-task-feasibility-20261102-attempt-01/transition-audit-train-smoke-v3.json`
has SHA-256
`b28f4f4c8d7b38c2c674f3535f93230f9eff28028576123318f99b40a5edaf6a`. The
source's `info.json` codec declaration is not treated as runtime truth; codec
is detected from the media stream. The smoke confirms decoding and annotation
availability only, not model quality.

The pinned VideoPrism checkpoint was also run on these three real frame windows
in the existing CPU-only JAX environment. The first compile+forward took
8.102 s; the next two took 1.436 s and 1.524 s (warm mean 1.480 s/window).
Output was finite `[1,8,768]`; peak working set was 4,182,413,312 bytes and
device was `TFRT_CPU_0`. This gives a local extraction-throughput measurement,
not the event count or end-to-end runtime for the full split. No GPU or weights
were changed. Two failed smoke attempts and the passing attempt are retained in
`videoprism-habit-smoke-attempt-log.json` (SHA-256
`31056ff418a70adfb59de865ec571f72db87690c3060a0636c76204dbf828ea8`).

## Full media and data audit

The pinned media transfer completed successfully: 816 train/validation files,
810 newly fetched (2,994,388,803 bytes) and six previously timed train files
reused only after size/SHA verification. Total selected media is 3,011,694,918
bytes; elapsed time was 2,194.7 seconds. No reserved media was requested or
loaded. C: had 2,942,636,032 bytes free after acquisition, above the retained
2 GiB reserve. The three-pair transfer timing file has SHA-256
`a2b22d52f731c3a58c2d580f168af0a4d4ee921226b907e0a964a8682960021f`.

The full media audit passed for all 408 selected episodes / 816 files. It
recomputed all file SHA-256 values, decoded 255,694 frames at 10 fps, and
confirmed exact video-frame/Parquet-row counts. It found 360 unique train and
48 unique validation video/Parquet assets, with zero train/validation overlap
in episode IDs, parent task groups, source input keys, video hashes, Parquet
hashes, or normalized option-text sets. It produced 936 eligible train and 144
eligible validation transition windows; no multi-option boundary was dropped
for an edge. Ninety-six train episodes in four whole groups have only one
distinct candidate action and contain no label transitions; they remain
integrity-audited but are ineligible for a choice task. Duplicate repeated
option descriptions were collapsed to their unique global action ID (744 train
and 32 validation occurrences). Validation still has only three task groups.

The external full-audit report is
`C:/CodexArtifacts/pretrained-reuse-r2/habit-task-feasibility-20261102-attempt-01/transition-audit-full-04.json`
with SHA-256
`61e7f7617de732df3c84146d1409de4bf7b337fde3c3f3295db9cf321adb3cd4`. The
executed auditor source SHA-256 is
`788103756cf331f26e97ba9034882db5172d2dc19744fc788a044abcfdb922e2`.
Two earlier full-audit attempts failed closed on duplicate repeated option
text and single-option episodes; the final audit explicitly handles/counts
both structural cases. These are eligibility corrections, not media integrity
failures.

Feature extraction/fitting is the next step, using only the frozen train and
three-group validation set. Report any metrics as exploratory and selection-
affected; this source cannot establish actor-independent or broad multimodal
quality.
