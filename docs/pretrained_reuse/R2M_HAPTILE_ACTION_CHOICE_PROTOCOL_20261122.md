# HapTile task-held-out coarse-action protocol — 2026-11-22

## Scope and status

This is a development protocol for one sensor-inclusive Decision task. It is
not a four-modality result, robot-success benchmark, or unseen-person estimate.
HapTile provides synchronized vision, bilateral tactile video, task
instructions, robot state and action trajectories, but no audio. The collection
uses one laboratory scene and does not document per-episode operator identity.
Claims from this source must therefore stay task-held-out and development-only.

The pinned Voxel51 conversion is revision
`c8593da98d3facb7a9293842514f0ee4ad3f2318` (CC BY 4.0), with 1,699 episodes
across 38 tasks. Seed 17 deterministically assigns whole task groups to 24
train, 4 calibration, 5 validation and 5 sealed-audit tasks. The full split,
including sealed task names, remains outside Git. Validation and calibration
are now accessed only through their public frozen task lists; sealed-audit
names, media, state, labels and features remain unopened.

## Frozen task

At a causal cutoff `t`, predict the dominant Cartesian direction of the
end-effector displacement over the next 250–500 ms. The seven fixed JSON
options are `-x`, `+x`, `-y`, `+y`, `-z`, `+z`, and `hold`:

```json
{
  "selected_id": "+x",
  "probabilities": {
    "-x": 0.0,
    "+x": 0.0,
    "-y": 0.0,
    "+y": 0.0,
    "-z": 0.0,
    "+z": 0.0,
    "hold": 0.0
  },
  "abstain": false
}
```

The values above are schema placeholders. For each episode, use one query at
the midpoint of the `/end-effector-pose` log-time range. Compute the current
position as the per-axis median over `[t-0.25 s, t]`; compute the target
position over `[t+0.25 s, t+0.50 s]`. If the Euclidean displacement is below
0.005 m, label `hold`; otherwise label the signed axis with the largest
absolute displacement. Ties resolve in x, y, z order. Future pose is target
only and must never enter model inputs. The exact label code is in
`src/tiny_omni_decision/haptile_action.py`.

Inputs end at `t` and cover only `[t-1.0 s, t]`: task instruction; front and
wrist RGB video; left and right tactile video; and robot state for a separate
state-only control. Use eight evenly spaced frames per stream. Do not include
future commands/states, haptic-feedback labels, task slug, episode ID, or
filename. Require all fixed windows and streams to be present and finite; do
not silently relabel missing data.

This is expert-action direction prediction, not proof that an action succeeds.
It does not claim that a language model can plan arbitrary robot actions.

## Train-only target preflight

The first proposed target was three-way gripper-command change. It was rejected
before validation inspection: a deterministic one-episode-per-train-task
sample (24 task groups, 89,798,329 bytes) produced 23/24 `hold` labels at the
proposed 0.10 threshold at episode midpoint. Across 9,688 overlapping
0.5-second command windows, only 968 (10.0%) crossed that threshold in either
direction; 8,720 were `hold`. Lowering the threshold did not resolve the
majority-class problem. The `command.plot.gripper` channel is therefore not
used as the target in this protocol.

The same 24 train-only episodes were then used to check the coarse 3-D action
label. All seven classes appeared at the fixed midpoint: `-x` 2, `+x` 4, `-y`
1, `+y` 3, `-z` 2, `+z` 7, and `hold` 5. This is only a feasibility signal.
The complete frozen train split now also has support for all seven classes:
`-x` 106, `+x` 190, `-y` 46, `+y` 86, `-z` 35, `+z` 246, `hold` 404
(1,113 episodes total). However, 19 of the 24 train task groups lack at least
one class. This is a material task-held-out generalization limitation; do not
balance by duplicating episodes or claim stable per-class performance from the
aggregate counts. Retain per-task and per-class counts in validation reports.

The 24 files were chosen one per train task by the minimum SHA-256 of
`haptile-train-preflight-v1|seed=17|<source path>`. Each was downloaded from the
pinned conversion revision and its byte count and LFS SHA-256 were verified.
MCAP CRC validation passed. The train-only acquisition receipt is outside Git
at `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-preflight-24x1\receipt.json`
(SHA-256 `d79ee622861b89a557320b2033ad42de2fd8de011afe5150063e01d53e72bedb`).
The target preflight is at `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-preflight-24x1\action-preflight.json`
(SHA-256 `d8175118772be4137265952848cf1c8f261b7d3d36c03dfdb4a653241320d3ed`).
The public task-split manifest SHA-256 is
`adacd722c1c51ce47bce139063dab1b01981d9e10d7e512c80e756e7d3e428e8`.

## Model comparison after train admission

Keep pretrained encoders frozen for the first pass. The planned Path A candidate
uses VideoPrism-B for visual and tactile streams, MiniLM-L3 for task
instructions, a small temporal state path, and a lightweight seven-way
Decision readout. Count every resident parameter. The source has no audio and
cannot establish four-modality readiness.

Compare on identical eligible episodes:

1. train-prior and instruction-only baselines;
2. robot-state-only;
3. front/wrist vision plus instruction;
4. tactile video plus instruction;
5. vision plus tactile plus instruction;
6. all preceding inputs with the state path.

Report sample and task-macro Accuracy, NLL, Brier and ECE, per-task counts,
class support, unique episodes, parameter count, package size, feature-extraction
cost, training time, inference latency, RAM/VRAM, and learning curves. Use
validation task-macro NLL for checkpoint selection and calibration groups for
temperature fitting. Compare paired intervals by task group. Include
option-order permutation, missing-stream, episode-shuffled tactile and
future-stream leakage controls. A fusion gain is credible only if it beats the
best unimodal counterpart on held-out task groups and interventions show the
prediction depends on the intended stream.

## Frozen split and access rules

The split hashes each task slug with
`tiny-omni-decision-haptile-task-v1|seed=17|task={task}`. All episodes and
modalities from one task stay together. The public manifest contains only
train/calibration/validation names and a commitment for sealed-audit tasks. The
private full manifest is external. Do not read sealed-audit files or labels.
Keep the frozen split, source revisions and pre-existing runs unchanged.

## Development split admission (2026-11-22)

The pinned conversion card was fetched from the source revision and verified
against SHA-256
`f0462ecc55d172e8a3c51ac5c874969e6f0482f093812482d28f730ba0558fbb`; its
38 task counts sum to 1,699 episodes. Validation and calibration were fetched
to separate external directories, with immutable receipts and no sealed data:

| Split | Tasks | Episodes | Bytes | Receipt SHA-256 | Action-preflight SHA-256 |
|---|---:|---:|---:|---|---|
| Validation | 5 | 169 | 724,644,960 | `58e931cc6c172e9fd7f7dd96c35b10ab30ef79e3675d552c1eb693ba50a2b5d4` | `8167c735eb16a0a7e809e74a6c63bafb420585db1433b9ef8bb974749ccce453` |
| Calibration | 4 | 169 | 583,592,810 | `92c4a86a6018bf32293963318adc9dfcc652908ab63a5e7c5ef805432145dd64` | `cc86a2833c8c6004ae78ca2dd6785b7315eb71e46b9ef27c5997f0be78d81ff7` |

Validation target counts are `-x:20,+x:20,-y:4,+y:26,-z:11,+z:24,hold:64`;
calibration counts are `-x:13,+x:19,-y:8,+y:22,-z:22,+z:26,hold:59`. These
small, task-held-out development sets are imbalanced and support only a pilot.
All 169 records in each split were hash-checked and CRC-validated before the
target preflight. Preflight outputs are external artifacts. Their task groups
are disjoint from train and each other by the frozen split; sealed task names
were not loaded. Validation metrics will select model arm and epoch;
calibration remains reserved for temperature fitting only.

A validation feature smoke on one fixed episode produced finite
`[1,4,8,768]` features in 25.2 seconds, using the pinned VideoPrism worker
environment on CPU. The first launch pointed at a bare uv Python without
NumPy and exited before model load; it wrote no features. The retry used the
already-installed VideoPrism virtual environment and passed. The full 169-row
validation feature extraction is now running in its own output directory.

## Implementation and local runtime evidence (2026-11-22)

The full train-only corpus is now present at
`C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-corpus-v1`:
1,113 MCAPs, 3,886,815,001 bytes. Every file matched its pinned LFS SHA-256 or,
for the one non-LFS file, its Git blob SHA-1; CRC checks passed. The receipt
SHA-256 is `f160edf505487cdf4333277c88987036b3d691eb25208aeb0a7f8445da13995c`.
The train-only action preflight SHA-256 is
`263ff3d1e7fde087632238d3b634629bc6315d203dfb8f3dcd225f26bb64daeb`. At the
time of that train preflight, no held-out media or labels were loaded. The
validation/calibration access recorded below occurred afterward; sealed audit
remains unopened.

The fixed four H.264 streams decoded in a 24-episode CPU preflight. A pinned
VideoPrism-B checkpoint (`be719a406d563b66f0ac969e7c94bab8e997c81a`, source
commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`) produced
finite `[24,4,8,768]` features from 96 clips in 176.0 seconds on CPU. This is a
throughput smoke, not a task-quality result.

Attempt `train-video-features-v1` encoded 54 examples before exposing a frame
selection defect: two evenly spaced target times could independently select
the same final frame. The underlying 1-second window had enough distinct frames.
The selector now performs an ordered, distinct-frame assignment minimizing
total squared timing error while preserving the same eight target times and
0.1-second per-frame bound. A regression test covers the exact failing
timestamps. A separate fixed 24-episode state/instruction smoke extracted all
25 causal robot-state channels and the `/instruction` text for every episode.
The state aligner uses as-of observations only and rejects stale channels.

An executable `HapTileActionDecisionScorer` now produces seven-way normalized
option JSON and supports fixed text-only, state+text, vision+text,
tactile-video+text, vision+tactile+text and full-input ablations. Actual pinned
MiniLM-L3 weights loaded on CPU and produced finite 384-wide embeddings. The
full candidate has 114,365,184 VideoPrism parameters, 17,389,824 MiniLM
parameters, and 519,683 decision-readout parameters (132,274,691 dense
parameters total); the separate text-only comparison head adds 196,865
parameters. These are parameter counts, not package-size or quality passes.
The selected full readout's option-order equivariance, JSON contract, temporal
gradient path and fixed parameter ledger are covered by CPU tests.

The current full-train VideoPrism extraction attempt is writing to
`C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-video-features-v2`.
At the 2026-11-22 09:12 JST snapshot it had completed 508/1,113 examples on CPU
(process 125676 remained live). The frozen experiment config SHA-256 is
`96a5e966ba6779d9d8e1cb4cb2f3aebf9b04cf0cc773519f603e17f4e812e766`. The
full extraction must finish and pass receipt/hash checks before model fitting.
No model has been trained; the downloaded validation/calibration data are held
for development-only evaluation and sealed-audit data remain unopened.

### Eligibility-gated feature run update (2026-11-22, 10:48 JST)

The earlier train VideoPrism attempts under `train-video-features-v1` and
`train-video-features-v2` remain preserved and incomplete. A label-blind fixed
window preflight now gates all feature paths without changing frame count or
timing: train has 1,112/1,113 eligible episodes, validation 169/169, and
calibration 169/169. The single train exclusion is
`data/put_stack_glass_cups/episode_024.fo.mcap`, which lacks eight distinct
causal video frames in each fixed stream. A complete label-blind state scan
also found that this same episode lacks pose history at the one-second context
start; it is not an additional exclusion. Train state features were therefore
re-extracted for the same 1,112 IDs, and their receipt/action/split provenance
and eligibility hash are recorded in the external result.

Fresh frozen text features are complete for train (1,112), validation (169),
and calibration (169), using the pinned MiniLM revision and weight hash above.
The 169-row validation feature join passed the exact eligibility, ID, media
hash, target, and causal timestamp checks. Calibration video features and the
train video feature extraction are still pending; the active CPU train-video
run `train-video-features-v3` had processed 396/1,112 examples at this snapshot.
No Decision fitting or calibration has run. The effective config SHA-256 is
`c0003bf98d8182e945fc45b25378f0bec5ba3b0889b7bdae89eafce08c284f80`; its
temperature rule fits a single deterministic scalar on calibration task-macro
NLL only after validation-based model selection. Calibration is not used for
arm or checkpoint selection. The full repository CPU suite passed with 3,258
tests and 11 skipped, and Ruff passed. Sealed audit remains unopened; no
quantization or product promotion has occurred.
