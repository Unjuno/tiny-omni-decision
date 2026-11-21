# HABIT source and task screen — 2026-11-21

## Decision

`METADATA_PREFLIGHT_PASS; QUALITY_ADMISSION_PENDING`. HABIT is an attractive
public source for a narrow human-aware video-plus-robot-state Decision pilot.
It is not a four-modality benchmark: the authors state that no audio is
recorded. This screen does not claim model quality, person-held-out
generalization, or product qualification.

## Source, rights, and people

The pinned Hugging Face dataset revision is
`bae97d251907743f8fb6db7e70c61d361df4dc1e`. Its card declares CC BY 4.0. The
paper says participants signed written consent covering public release under
CC BY 4.0, and the public videos blur faces. It also states that body shape,
clothing, posture, and gestures remain visible because they carry the intended
signals. No audio is collected. The paper reports approximately ten human
collectors at one institution, so this data cannot establish broad
unseen-person generalization. The public episode metadata has no participant
identity field; `unit_name` is not treated as a person ID.

The official release describes 10,563 episodes (164.19 hours), 60 tasks, five
synchronized RGB views, robot kinematics, and human/robot subtask annotations.
This supports a bounded Decision formulation: given a causal video window,
current robot state, task instruction, and the task's fixed candidate
subtasks, return calibrated probabilities over the next robot subtask. The
target is `low_level_task_index`; the option text comes from the corresponding
task's subtask inventory. Training and evaluation must exclude action fields,
future frames, and future state. Required controls are text/options-only,
state-only, video-only, and paired input, plus shuffled/missing-input tests.

## Metadata and schema audit

Only public metadata, task descriptions, and one sample-config Parquet episode
were downloaded to
`C:\CodexArtifacts\pretrained-reuse-r2\habit-source-screen-20261121-attempt-02`.
The only videos acquired were train episode 0 and development-validation
episode 1802, one `front_view` clip each (9,269,185 bytes total), saved in
`C:\CodexArtifacts\pretrained-reuse-r2\habit-media-runtime-smoke-20261121-attempt-01`.
Only those two full-split Parquet episodes were opened for target/option schema
mapping, in
`C:\CodexArtifacts\pretrained-reuse-r2\habit-label-schema-20261121-attempt-01`.
The 60-episode sample Parquet was used only for schema/index inspection. No
sealed-audit episodes, labels, or media were accessed. The source revision and file hashes are in
[`habit-source-screen-20261121.json`](../../manifests/pretrained_reuse/habit-source-screen-20261121.json).

## Runtime and target-mapping smoke

PyAV 18.1.0 decoded all 653 frames (episode 0) and 469 frames (episode 1802)
at 10 fps and 640×480; timestamps were monotonic and neither clip had an
audio stream. The pinned VideoPrism-B checkpoint and source tree were loaded
from the existing R1 CPU environment without changes. Eight uniformly spaced
frames per clip produced finite `[1, 2048, 768]` features and pooled 768-D
vectors. Combined wall time was 15.822 s including the first JAX compile; peak
working set was 4,157,022,208 bytes. Both clips were processed on `TFRT_CPU_0`;
the GPU was not used. Feature hashes and per-video timings are in the local
`videoprism-runtime-smoke.json` receipt.

The two corresponding full Parquet episodes contained 653 and 469 rows,
respectively. All observed `low_level_task_index` values mapped exactly to the
episode's candidate `low_level_tasks` (three choices for train episode 0 and
four for validation episode 1802). `observation.state` is a 14-D vector. The
Parquet also contains action fields; the proposed model input excludes them.
This confirms the schema path only, not that video/state improves decisions.
See [the fixed mapping smoke](R2_HABIT_DATA_MAPPING_SMOKE_20261121.md).

The pinned metadata contains 10,563 unique episode indices and input keys, 63
`task_index` values, 66 `sid` values, and 63 distinct high-level instructions.
`full/meta/info.json` and `tasks.jsonl` report 63 tasks, while the paper/card
report 60 and the pinned `task_details/` directory contains 60 Markdown files.
Those files encode 65 scenario-ID tokens; `S7791` has no matching task detail.
Two task-detail documents contain a spelling variant `Colaborator`, which the
preflight parser normalizes. Exact repeated instruction text connects task
records with conflicting role labels; such connected groups are kept
train-only. These differences must remain visible in the report and must not
be “fixed” by guessing.

The official page reports 3,198 Collaborator, 3,969 Coworker, and 3,396
Supervisor episodes. Assigning roles through the task-detail files yields
3,229 Collaborator, 3,824 Coworker, 3,396 Supervisor, and 114 episodes under
the undocumented `S7791`. The totals agree, but role counts do not. The split
preflight therefore quarantines the 114 undocumented episodes and keeps the
two mixed-role task groups in train only.

## Proposed deterministic development split

`src/tiny_omni_decision/habit_preflight.py` connects records when they share an
official task-detail SID group, `task_index`, or exact NFKC/case/whitespace
normalized high-level instruction. Pure-role groups are stratified with seed
17; ambiguous role groups stay in train; the undocumented SID is quarantined.
This yields 40 train groups / 7,655 episodes, 9 validation groups / 1,458
episodes, 9 sealed-audit groups / 1,336 episodes, and 1 quarantined group / 114
episodes. Validation and sealed audit each contain three groups from each
role. Episode-ID hashes are fixed in the machine-readable preflight receipt.
Audit episode IDs are intentionally neither written nor returned by the
normal preflight path. The source has no publisher validation/test split; the
derived sealed audit is therefore a new task-group holdout, not a publisher
test set.

This split is not yet approved for quality fitting. The media bytes have not
been acquired, the 63-vs-60 task discrepancy is unresolved, and actual video
asset uniqueness/content hashes have not been verified. Later media
preparation must also prove that every episode maps to exactly five valid
views and that no episode or task group crosses splits. Any video duplication
outside these metadata identities remains unknown until media-level checks.

## Next bounded work

1. Resolve `S7791` against the pinned public task metadata or keep it
   quarantined.
2. Confirm full-task options and target coverage from train-only episode
   metadata without opening audit labels.
3. Freeze the task groups, episode IDs, evaluation controls, and a small
   train/validation media subset before video acquisition.
4. Audit the planned train/validation media subset and source identities before
   scaling feature extraction. No GPU work was started by this screen.

## Evidence

- [Official HABIT project page](https://habit-dataset.github.io/)
- [Pinned dataset card](https://huggingface.co/datasets/configinc/HABIT/tree/bae97d251907743f8fb6db7e70c61d361df4dc1e)
- [Pinned dataset README](https://huggingface.co/datasets/configinc/HABIT/blob/bae97d251907743f8fb6db7e70c61d361df4dc1e/README.md)
- [HABIT paper, v2](https://arxiv.org/abs/2606.31682)

No cloud compute was used. Existing artifacts and other candidate datasets
were not modified.
