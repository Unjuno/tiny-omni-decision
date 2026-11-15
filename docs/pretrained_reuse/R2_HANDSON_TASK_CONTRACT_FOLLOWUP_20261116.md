# HandsOn task-level contract follow-up — 2026-11-16

## Decision

Keep HandsOn `REVIEW_NOT_ADMITTED` for a model-quality run. The pinned
release's session-disjoint split is intact, but its individual source tasks
have only 1–9 validation sessions (median 2; reverified below). The
task-level counts below confirm that the promising ball, cup, and block
examples do not approach the roadmap's provisional 200 independent-group
floor. Their targets and scene contracts also differ, so pooling them would
not produce one coherent Decision task. Do not acquire trajectory/media
payloads, fit a model, or evaluate quality from this screen.

This follow-up also separates the simulator's task-specific observation API
from the released dataset's public 52-dimensional state. It does not infer
that simulator-only target fields are present in the dataset.

## Pinned inputs and method

- HandsOn dataset revision:
  [`7e19e2675464ebff9742923b20267e5af5ad645e`](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/tree/7e19e2675464ebff9742923b20267e5af5ad645e).
- Source code revision:
  [`c9598b53fc2a01355e74837de397bd7b2b1ab602`](https://github.com/TsinghuaC3I/HandsOn/tree/c9598b53fc2a01355e74837de397bd7b2b1ab602).
- Dataset schema: pinned [`meta/info.json`](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/blob/7e19e2675464ebff9742923b20267e5af5ad645e/meta/info.json), 3,582 bytes, SHA-256 `726c2bb13911ac50223cb91f85559ee15362aae2bb628ca31dad38de841b0671`.
- Split/task metadata: pinned [`source_manifest.jsonl`](https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/blob/7e19e2675464ebff9742923b20267e5af5ad645e/source_manifest.jsonl), streamed in memory, 42,610,607 bytes, SHA-256 `6cced3356edb916038d160c3262238d0bc606ef678885b510db20b9dea69a942`.
- Exact task contracts: [t24](https://github.com/TsinghuaC3I/HandsOn/blob/c9598b53fc2a01355e74837de397bd7b2b1ab602/handson/handson/tasks/t24_place_ball_in_bowl/config.py), [t27](https://github.com/TsinghuaC3I/HandsOn/blob/c9598b53fc2a01355e74837de397bd7b2b1ab602/configs/multi_task/t27_place_largest_fitting_ball.yaml), [t11](https://github.com/TsinghuaC3I/HandsOn/blob/c9598b53fc2a01355e74837de397bd7b2b1ab602/handson/handson/tasks/t11_line_up_blocks/config.py), and the [t24 object XML](https://github.com/TsinghuaC3I/HandsOn/blob/c9598b53fc2a01355e74837de397bd7b2b1ab602/handson/handson/sim/envs/xmls/ball_to_bowl_objects.xml) plus [asset map](https://github.com/TsinghuaC3I/HandsOn/blob/c9598b53fc2a01355e74837de397bd7b2b1ab602/handson/handson/sim/envs/xmls/a_series_ycb_mesh_assets.xml).
- The aggregate selected only `split`, `collection_session_id`, and `source_task`; output contains task-level episode and distinct-session counts only. Prompts, record IDs, state/action values, annotations, media paths, and evaluation labels were not analyzed or emitted. No dataset payload, model, or simulator was opened.

## Public state schema versus simulator observations

The pinned dataset schema declares `observation.state` as float32 shape `[52]`,
with `state_semantics: current_measured` and per-side layout
`palm_xyz_m3`, `palm_rotvec_rad3`, and `hand_control_rad20`. It does not
declare per-object poses, target identity, object class, or task-specific
success metrics in that vector.

By contrast, the pinned simulator task config for `t24_place_ball_in_bowl`
adds `target_ball_pose`, each ball pose, `bowl_pose`, `target_id`, and
`target_name` to its task observation adapter. Those simulator-only keys are
not evidence that the released 52D demonstration state contains the same
fields. An evaluation must consume the pinned dataset schema or explicitly
declare a different simulator observation contract; these two interfaces must
not be conflated.

## Task-level split counts

Counts come from the pinned source manifest and use distinct
`collection_session_id` values as groups; no episode was treated as an
independent replicate.

| Source task | Train episodes / sessions | Validation episodes / sessions | Result |
|---|---:|---:|---|
| `a03_ball_to_bowl` (config `t24_place_ball_in_bowl`) | 438 / 18 | 62 / 3 | Too few independent validation sessions |
| `ycb02_select_largest_fitting_ball` (config `t27_place_largest_fitting_ball`) | 442 / 60 | 58 / 6 | Too few groups; additionally blocked by the task's CC BY-NC asset lineage below |
| `c12_push_blocks_to_line` (config `t11_line_up_blocks`) | 272 / 25 | 28 / 3 | Too few independent validation sessions |
| `b320_cup_stabilized_faucet_open` | 196 / 16 | 104 / 7 | Too few independent validation sessions |

The reaggregated source manifest has 895 train sessions and 155 validation
sessions, with zero session overlap. Across all 54 source tasks, validation
session counts have minimum 1, maximum 9, and median 2. A cross-task aggregate
would mix different action semantics, objects, task prompts, and success
conditions, and would not support a per-task quality claim. This screen does
not treat the provisional group floor as a substitute for a formal power
calculation; with at most nine held-out sessions per task, no current task
supports a meaningful grouped paired-control evaluation.

## Rights and both-input requirement

The pinned `t24_place_ball_in_bowl` object XML resolves the bowl, tennis ball,
golf ball, and mini-soccer ball meshes/textures to
`lightwheel_ycb/`. The upstream
[Lightwheel-YCB README at commit
`03312c5d12b75e70c692fa341d683f5aabab1d99`](https://github.com/LightwheelAI/Lightwheel-YCB/blob/03312c5d12b75e70c692fa341d683f5aabab1d99/README.md)
declares CC BY-NC 4.0. The pinned README was fetched through Git and the
license wording was verified at that exact commit. This is not compatible
with the product-directed model path; no t24 media or object payload was
acquired. The exact t27 task family
was already traced to the same non-commercial asset source in the parent
[source/task screen](R2_HANDSON_SOURCE_TASK_SCREEN_20261116.md).

No typed Decision target and matched controls were frozen. In particular, the
screen does not establish that video and measured robot state are both needed,
nor does it establish how a target ball is represented by the released task
instruction. No video-only, state-only, prompt-only, shuffled, missing-input,
temporal-offset, or candidate-order evaluation was run.

## Reproduction boundary and next gate

- Source-manifest byte count and SHA-256 match the parent screen's pinned
  receipt. Counts above were computed without writing the manifest to disk.
- The separate code-only checkout is at
  `C:\CodexArtifacts\pretrained-reuse-r2\handson-code-screen-20261116`,
  with `FETCH_HEAD` exactly `c9598b53fc2a01355e74837de397bd7b2b1ab602`.
- No media, trajectory table, evaluation label, checkpoint, feature cache,
  sealed audit, or user dataset was opened; no GPU or training was used.
- Reconsider HandsOn only if a product-compatible task asset set is traced,
  one coherent Decision target has a defensible held-out group/power plan, and
  the target demonstrably requires both video and state under matched controls.
  Do not solve the current gap by random repartitioning sessions or pooling
  unlike tasks.
