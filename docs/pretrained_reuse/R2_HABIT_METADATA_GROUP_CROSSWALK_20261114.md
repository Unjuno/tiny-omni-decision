# HABIT metadata group crosswalk — 2026-11-14

## Disposition

Keep `configinc/HABIT` **`REVIEW_NOT_ADMITTED`** for a product-quality R2
decision task. The pinned source metadata contains at most 60 task-document
groups, below the provisional minimum of 200 independent source groups. It
does not expose participant or collection-session identities needed to prove
that scenario groups are independent. The published count of 308 unique
human–robot subtask pairs is a count of task-pair identities, not independent
examples or source groups.

This is a metadata-only admission finding, not a model-quality failure. No
source data were fetched in this follow-up, no video or Parquet payload was
opened, no new split was made, and no model, features, predictions, or GPU were
used. The prior 408-episode feasibility pilot and its validation remain
preserved; its validation labels/transition counts were already inspected and
are not a blind evaluation set.

## Pinned source and inspected metadata

- Dataset: `configinc/HABIT`
- Hub revision recorded by the existing metadata artifacts:
  `bae97d251907743f8fb6db7e70c61d361df4dc1e`
- Existing source metadata directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\habit-source-screen-20261102-attempt-01\source-metadata`
- Existing feasibility directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\habit-task-feasibility-20261102-attempt-01`
- The official README describes `sid` as a **scenario ID** linking an episode
  to task documentation. Multiple SIDs can describe execution variations of
  the same underlying task; some are grouped into a single task-details file.
- The metadata contains 10,563 episode rows, 66 distinct SIDs, and 63 distinct
  `task_index` values. The README describes 60 tasks and the source snapshot has
  60 task-details files. The small discrepancy among task counts is unresolved;
  none is counted as an independent-person or independent-session count.
- All 10,563 episodes are declared in the source metadata as one `train` range
  (`0:10563`); there is no publisher-provided validation/test partition.
- Episode metadata fields include `sid`, `task_index`, instruction/task fields,
  and `input_key`; they contain no participant ID, collection-session ID, or
  recording-group field. The 182 human-subtask rows and 157 robot-subtask rows
  do not add participant/session identity. Pair/task counts cannot substitute
  for that missing grouping.

## Existing pilot evidence and limits

The preserved feasibility plan split whole task-details filenames, treating
combined-SID filenames as indivisible. It contains 15 train groups, 3
validation groups, and 3 reserved group names; the reserved selection has zero
episodes. The selected train and validation portions contain 360 and 48
episodes respectively. The existing transition audit reports 936 train and
144 validation eligible boundaries, with zero overlap for video hashes,
Parquet hashes, input keys, and option-text sets **within that selected pilot**.
This does not establish participant/session independence or provide 200
independent groups. The existing validation was already inspected for target
and transition feasibility, so it must not be re-used for model selection.

## Reproducibility hashes

SHA-256 values below identify the already-local inputs used for this
crosswalk; the source metadata revision above identifies their upstream pin.

| File | SHA-256 |
|---|---|
| `README.md` | `44B6BD40BEBAEE29A2A22BA023BBC8B811DF9DB4158B6EF9C219A9B669223470` |
| `full__meta__episodes.jsonl` | `0AA53B5AB7C6C22ED8662BC77CE0BA3D5BCFA62BC885F1A928C586F76E95FFF5` |
| `full__meta__human_subtasks.jsonl` | `6F4BA296CA2A79423D9AB28119A52BE01402BBF08C856BFF20A8D40A064C3877` |
| `full__meta__subtasks.jsonl` | `DCB0A7FE179C3F015BCEF5BF3047DE34D92DAB2F0B8E2A55521B2FB0A21203A9` |
| `full__meta__tasks.jsonl` | `9D407780B6AE0AE31677556BFD97982B74339389D52ECBE9E3A7C6A2948BE013` |
| `full__meta__info.json` | `41749996D359830EDE789179D4BB0C08A90DB1A3BD97928E0F7D8D05F948CE08` |
| `metadata-plan.json` | `FE9D8012131DB25578D25A6501A9C56C50DD2D4DDEF33CFD2145D3AA81CB8F1B` |
| `transition-audit-full-04.json` | `61E7F7617DE732DF3C84146D1409DE4BF7B337FDE3C3F3295DB9CF321ADB3CD4` |

The independent read-only summary counted JSONL rows, unique `sid` and
`task_index` values, field names, and the `sid`↔`task_index` mapping. No raw
identity values or episode content are reproduced here.

## Next action

Do not expand the HABIT pilot, acquire additional HABIT media, or fit a readout
for the provisional R2 quality gate. Continue source admission for a fresh,
rights-cleared task with a valid independent group unit and at least 200 groups;
only then freeze the paired unimodal, missing, shuffled, stale/time-offset, and
contradiction controls before any fusion fit.
