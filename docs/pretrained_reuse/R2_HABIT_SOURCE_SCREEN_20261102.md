# HABIT source and task screen — 2026-11-02

## Disposition

**`REVIEW_NOT_ADMITTED` for R2-M quality evaluation.** HABIT is a promising,
rights-cleared source for a narrow synchronized RGB-video plus robot-state
decision task: the release contains co-present human/robot interaction, five
camera streams, bimanual robot state/actions, and gesture/collaboration tasks.
The current metadata does not provide participant IDs or a publisher
train/validation split. It provides 66 scenario (`sid`) groups and 63 task
indices, but those do not establish actor-independent or scene/layout-held
evaluation. The sample configuration is only one episode per task and is not
a split. No task protocol has yet shown that both video and proprioception are
needed for one fixed-option decision. No model, media, or episode state/action
payload was downloaded or evaluated.

## Pinned source and rights

- Dataset: [`configinc/HABIT`](https://huggingface.co/datasets/configinc/HABIT),
  pinned Hub revision
  `bae97d251907743f8fb6db7e70c61d361df4dc1e` (Hub API SHA; public, ungated).
- The dataset card declares CC BY 4.0. The author paper states that participants
  consented to public CC BY 4.0 release and that refusal/withdrawal is allowed.
  It also documents that faces are blurred in public videos, while body,
  clothing, posture, and gestures remain visible; no audio was recorded. A
  right to withdraw persists after publication, with removal from
  author-controlled mirrors, but previously downloaded copies cannot be
  recalled. These conditions make the stated license compatible at the
  declaration level; any future use must avoid identity inference and retain
  attribution.
- The author paper and project page report 10,563 episodes, 164.19 hours, 60
  high-level tasks, three interaction roles, five synchronized RGB cameras,
  and a bimanual Franka robot. The paper describes approximately ten human
  collectors from one institution, which limits actor diversity.
- The pinned `full/meta/info.json` declares 10 fps and `train: 0:10563`; its
  `robot_type` field says `human`, which conflicts with the paper's robot
  description and should be clarified before using it as a runtime schema
  contract. The metadata exposes state, action, video, and annotation groups;
  it contains no audio modality.

Primary sources: [project page](https://habit-dataset.github.io/),
[pinned Hugging Face dataset](https://huggingface.co/datasets/configinc/HABIT/tree/bae97d251907743f8fb6db7e70c61d361df4dc1e),
[author paper and ethics statement](https://arxiv.org/html/2606.31682v1).

## Metadata-only inventory

At the pinned revision, the Hub API reports 250,134,941,418 bytes of stored
content. The dataset card reports 253 GB for `full` and approximately 1 GB for
`sample`; `sample` contains 60 episode records, with 58 unique task indices and
60 scenario IDs. It is a schema/runtime example, not a development split.

Only these eight metadata files were fetched from the pinned revision (total
9,446,061 bytes). No Parquet, MP4, checkpoint, or evaluation payload was
fetched:

| Pinned file | Bytes | Local SHA-256 |
| --- | ---: | --- |
| `README.md` | 15,540 | `44b6bd40bebaee29a2a22ba023bbc8b811df9db4158b6ef9c219a9b669223470` |
| `full/meta/episodes.jsonl` | 9,311,649 | `0aa53b5ab7c6c22ed8662bc77ce0ba3d5bcfa62bc885f1a928c586f76e95fff5` |
| `full/meta/tasks.jsonl` | 4,393 | `9d407780b6ae0ae31677556bfd97982b74339389d52ecbe9e3a7c6a2948be013` |
| `full/meta/subtasks.jsonl` | 16,170 | `dcb0a7fe179c3f015bcef5bf3047de34d92dab2f0b8e2a55521b2fb0a21203a9` |
| `full/meta/human_subtasks.jsonl` | 18,145 | `6f4ba296ca2a79423d9ab28119a52be01402bbf08c856bff20a8d40a064c3877` |
| `full/meta/info.json` | 19,105 | `41749996d359830ede789179d4bb0c08a90db1a3bd97928e0f7d8d05f948ce08` |
| `full/meta/modality.json` | 7,624 | `97d9d99879698a21ef7382c9733703377ebcb9ebcd35f73f06586637ca574131` |
| `sample/meta/episodes.jsonl` | 53,435 | `41c772fc24570ea6f50717f949657e89c9e0f76151bfbedd35b055470dc82ce4` |

The local copy is in
`C:/CodexArtifacts/pretrained-reuse-r2/habit-source-screen-20261102-attempt-01/source-metadata/`.
It is an external audit artifact, not committed to Git.

The pinned `task_details/` directory was also fetched as text-only metadata:
60 Markdown files, 73,824 bytes total. Every local file matched the pinned
file-tree byte count and SHA-256. The inventory file is stored at
`C:/CodexArtifacts/pretrained-reuse-r2/habit-source-screen-20261102-attempt-01/task-details-file-inventory.json`
with SHA-256
`5c69012f32394f77d5e783e7557a3e58ca424f6659efe4c504ba947989b23458`.
The 60 detail files account for 65 of the 66 `sid` values by their filename
references; `S7791` has 114 episode rows (`task_index=54`) but no matching
detail filename. Parsed top-level role headings in the detail files are
8 `Colaborator`, 12 `Collaborator`, 19 `Coworker`, and 21 `Supervisor`;
normalizing the apparent spelling variant gives 20/19/21 files. This differs
from the paper's stated 20 tasks per role and is unresolved. These details do
not establish participant- or physical-scene-independent groups.

## Task and split feasibility

- The pinned full episode metadata has 10,563 unique episode indices, 66 `sid`
  values, and 63 task-index values. `sid` is documented as a scenario ID; it
  can prevent those recorded scenarios from crossing a custom split, but it is
  not a verified physical scene or participant identifier.
- Episode metadata has no actor, person, participant, scene, layout, or
  environment ID field. `unit_name` has four collection-unit values, not
  per-person IDs. The paper reports about ten collectors but does not map them
  to episode rows in the released metadata.
- The only publisher split is a single `train` range over all 10,563 episodes.
  No official validation or audit partition was verified. A custom `sid`-held
  split is possible from metadata, but would still mix participants and
  physical recording environment; report only as scenario-held development
  evidence if ever used.
- The strongest candidate decision direction is future robot-action choice
  during a human-aware subtask, using synchronized video and only current/past
  proprioception at inference. Before any media acquisition, freeze one task,
  candidate construction, timestamp semantics, split IDs/order, and matched
  video-only / state-only / joint / missing / temporally misaligned controls.
  Exclude future state/action, task/subtask annotations, and episode keys from
  model inputs. The source's gesture tasks make video relevant, but do not by
  themselves establish that joint video+state is necessary or that the
  readout generalizes to unseen people/scenes.

## Resource gate and next step

At screen time C: had about 7.08 GiB free. The card's 1 GB `sample` config
could fit, but has too few episodes and no split; the full release is about
233 GiB by the Hub's exact byte count. No media was fetched. Before any media
request, use the metadata-only file tree to calculate exact bytes for a
predeclared episode-group subset and decide whether a fresh `sid`-held
development task has enough independent support. If not, stop this candidate
without downloading media. Keep its status `REVIEW_NOT_ADMITTED` until the
task, split, grouping, input-necessity, and local-storage gates pass.
