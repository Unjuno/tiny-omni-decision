# HandsOn synchronized video/state source and task screen — 2026-11-16

## Disposition

**`REVIEW_NOT_ADMITTED` for model fitting or quality evaluation; retain as the
highest-priority bounded R2-M follow-up.** This is a materially stronger
candidate than the recent MEVA and AIST screens: the publisher exposes a
session-grouped train/validation split and the released records contain
synchronized multi-view RGB, robot state/action, and task instructions. The
official validation has only 155 independent collection sessions overall,
and 1–9 sessions per source task (median 2). This is below the roadmap's
provisional 200-group floor and cannot support per-task quality claims. The
dataset is simulated, and the selected task's third-party asset rights plus a
Decision target that demonstrates both video and state necessity still need
to be established. No video, image, trajectory table, model, or evaluation
label was downloaded or opened.

## Pinned source evidence

- Dataset card revision `7e19e2675464ebff9742923b20267e5af5ad645e`:
  <https://huggingface.co/datasets/Cyan666/HandsOn-Datasets/tree/7e19e2675464ebff9742923b20267e5af5ad645e>.
  The card says 18,000 simulated episodes, 15,178 train episodes from 895
  collection sessions, 2,822 validation episodes from 155 sessions, three
  synchronized RGB cameras, 52-dimensional robot state/action, and task
  prompts. It states the dataset and reader are MIT, while third-party assets
  retain their own licenses. The card quotes approximately 111 GB for its
  18,000-episode PhysX/Wuji1 corpus. The same revision also contains a separate
  10,000-trajectory Newton/Wuji2 addition and the Hub repository footer reports
  304 GB total; do not treat 111 GB as the size of every file in the revision.
- Pinned split metadata `meta/splits.json`, HTTP 200, 187,400 UTF-8 bytes,
  SHA-256 `8c4de928722371acf2806d9826751dce37a73443712748d779b4715155913cf7`.
  It identifies seed `v16_session_split_20260916_v1`, groups by collection
  session (unioned across verified aliases), and lists 15,178 train / 2,822
  validation episode indices. Its prior-exposure field says these trajectories
  existed in a V15 training universe; the split is only held out for a fresh
  V16 run and is not unseen to models already trained on V15.
- Pinned `source_manifest.jsonl` at the same dataset revision: 42,610,607
  bytes, streamed in memory (not saved), SHA-256
  `6cced3356edb916038d160c3262238d0bc606ef678885b510db20b9dea69a942`.
  Aggregating only `split`, `collection_session_id`, and `source_task` found
  18,000 records, 895 train sessions, 155 validation sessions, zero session
  overlap, and all 54 source tasks in both splits. Across those source tasks,
  validation session counts range from 1 to 9 (median 2). No record IDs,
  prompts, state/action values, or labels were emitted by the aggregate.
- Official code repository pinned at `c9598b53fc2a01355e74837de397bd7b2b1ab602`
  (tree `84f99f1f5438384b1e1faab6ff1a8aeccc9b6a23`):
  <https://github.com/TsinghuaC3I/HandsOn/tree/c9598b53fc2a01355e74837de397bd7b2b1ab602>.
  Root `LICENSE` is MIT. `NOTICE` says task designs/assets reuse material from
  DexJoCo and DexVerse, and robot descriptions, vendored assets, and NVIDIA
  Isaac Sim/Isaac Lab retain separate terms. Therefore root MIT alone does not
  clear every rendered task's assets for product-directed model training.
- The selected task has not been frozen. As one task-design example only,
  pinned `t24_place_ball_in_bowl/config.py` includes target-ball and bowl state
  fields in its simulator observation contract; those fields could make a
  state-only baseline trivially strong. The dataset card's 52D state schema is
  different from this simulator observation contract. Do not assume the
  released trajectory state contains the same target fields.
- The paper describes a simulated manipulation benchmark and reports 29.1%
  task-macro success for its best multi-task policy. That is a distinct
  closed-loop success metric, not a comparable option Accuracy threshold:
  <https://arxiv.org/abs/2610.26765>.

## R2-M assessment

| Gate | Finding | Disposition |
|---|---|---|
| Rights | Dataset card says MIT; pinned NOTICE preserves third-party robot/task/object and simulator terms | Screen one task's complete rendered-asset lineage before acquisition or fitting |
| Synchronized streams | Three RGB views plus 52D state/action are described and the source manifest has per-episode media hashes | Promising; no payload integrity replay performed |
| Independent split | Official session grouping; zero session overlap confirmed in source manifest | Good split construction, but only 155 validation groups overall and 1–9 per task |
| Prior exposure | Dataset card says data existed in V15 training universe | Any V15-trained checkpoint is ineligible; exact pretrained encoder overlap is still unknown |
| Decision target | Upstream supplies demonstrations, not a typed supplied-option Decision task | Must define a derived candidate-action choice task before fitting |
| Both-input necessity | No matched controls exist; state/action or task prompt may reveal the target | Not established |
| Product relevance | Simulated bimanual manipulation with RGB and robot state | Useful bounded sensor-inclusive decision probe, not real-world or four-modality evidence |
| Local feasibility | The 18K PhysX/Wuji1 card says about 111 GB; the full Hub revision reports 304 GB including a separate raw addition. Selected-task archive, video, and table sizes were not summed. | Do not download until task/rights/split freeze and an exact C-drive size estimate |

## Cheapest decisive next step

Before any media acquisition, choose one action-selection task and use only
the pinned code/config/provenance to:

1. trace every object, robot, and renderer asset in that task to explicit
   compatible terms;
2. confirm the task's session counts in both splits and determine whether a
   cross-task aggregate is necessary to obtain useful held-out group coverage;
3. predefine a text-free or instruction-conditioned typed choice over
   recorded next-action chunks, with hard negatives fixed from training data;
4. compute option-only, prompt-only, task-ID-only, and sensor-only
   predictability from training metadata, without scoring validation labels;
5. specify video-only, state-only, joint, shuffled, missing-input, temporal
   offset, and candidate-order controls plus a session-bootstrap power plan.

If a single task cannot reach the provisional validation-group floor, do not
shrink or randomly repartition its validation sessions. Either establish a
predeclared cross-task decision with held-out sessions and task-macro
reporting, or leave this source unadmitted. Do not evaluate the official
Transfer-16 or any final/audit set during candidate design.

## Reproduction and boundary

- Repository/base commit: `bfb13bf6e9ac4f2ce244eb51e470afa907587380`.
- Source-code commit/tree and dataset revision are pinned above. The 187.4 KB
  split JSON was read. The 42.6 MB source manifest was streamed twice (once
  for count aggregation, once for SHA-256 plus the same aggregate), then
  discarded; an additional first-line schema probe emitted key names and
  value types only.
- No media, parquet trajectory, evaluation label, checkpoint, feature cache,
  sealed audit, or simulator was opened or run. No acquisition, training,
  model inference, or GPU work occurred.
- `REVIEW_NOT_ADMITTED` is a source/task-screen outcome, not a claim that the
  data are unusable or a quality result.
