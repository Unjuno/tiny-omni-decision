# R2 multimodal task admission matrix — 2026-11-04

## Purpose and scope

Metadata-only reconciliation of the existing tracked R2 source screens,
protocols, and result reports. No source media, model weights, feature arrays,
sealed-audit records, or the two untracked PhysicalAI split manifests were
opened. This matrix does not admit a task or authorize another fit.

The R2-M gate requires rights-cleared data, an auditable independent group
split, a frozen task where both proposed streams are needed, matched unimodal
and input-perturbation controls, and sufficient complete data. “Both inputs
appear in the task definition” is not empirical evidence that both are useful.

## Candidate disposition

| Candidate | Rights / pinned source | Task and groups | Existing evidence | Data/cache and exposure | Decision |
|---|---|---|---|---|---|
| [RoboMNIST AV repetition match](R2_ROBOMNIST_AV_MATCH_PREFLIGHT_20261103.md) | CC BY 4.0, official Figshare v1; source/article pins in `R2_ROBOMNIST_SOURCE_20261014.md` and `R2_ROBOMNIST_EPISODE_TASK_FEASIBILITY_20261103.md` | Frozen whole-repetition AV match; 780 train / 1,000 validation groups, pair-unit split. Episode-level AV synchronization is supported; frame-level alignment is not. | No full validation score yet. The task is structurally correspondence-dependent, but no measured joint-vs-unimodal benefit exists. | Feature cache is 1,778/1,780; both missing groups are validation-only. This is previously exposed development metadata, not blind or scene-held-out. Cross-task metadata overlap is documented. Extraction requires the original 2 GiB free-space guard. | **Closest viable next evaluation, currently blocked by two missing features and disk reserve.** Do not score 998 groups or refit. |
| [DCASE 2025 / STARSS23 AV match](R2_DCASE2025_AV_MATCH_CV_20261101.md) | STARSS23 v1.1 official MIT release; bounded source audit at `R2_STARSS23_SOURCE_20261011.md` | Four-choice recording identity match; grouped by 78 parent recordings and 9 rooms. | Nested-room CV: joint 26.56% vs 25% chance; NLL/Brier worse than uniform; paired intervals include zero. Direct cosine: 27.34%, uncertainty wide. | Existing frozen development features; public dev-test and sealed audit not used in these reports. Source sample is small and room clusters are few. | **Do not tune or repeat. Weak/negative evidence; does not pass both-input quality gate.** |
| [Eidon Tracker POV + IMU](R2_EIDON_TRACKER_POV_MATCH_20261029.md) | Pinned video and IMU revisions; cards report CC BY 4.0. Rights do not remove privacy concerns in unblurred private-home footage. | Video-to-sensor recording match; contributor/task grouping. | Pooled and LSTM: 7/24 (29.2%) on 3 held-out contributors; reversed-time LSTM was slightly better. Strong overfit; no useful temporal or both-input effect. | Tiny, already exposed development set. | **Do not repeat or promote. Insufficient independent groups and negative control evidence.** |
| [NVIDIA PhysicalAI SingleArm](R2_PHYSICALAI_FRESH_ACTION_PREFLIGHT_20261103.md) | Publisher revision `73c00c18e7ca108e91846294b20a1d7cd5de9340`, CC BY 4.0 per tracked source screen. | Video + proprio action-prototype prediction; existing episode split. No scene/layout identity, and review found shared simulated workspace. | Existing pilot: joint 72.5% vs proprio 70.4%, but episode-bootstrap interval includes zero; heads reached 100% train accuracy and calibration is mixed. This is not robust both-input benefit. | Fresh split metadata exists in tracked report, but acquisition attempt stopped at HTTP 429 with 175 files; current free space is below reserve. The 1,000 reserved IDs remain unopened. Do not read user-owned untracked manifests. | **Not admitted. Scene independence and reliable complementarity remain unproven; no acquisition or refit now.** |
| [HiFi-UMI-2K](R2_HIFI_UMI_2K_SOURCE_SCREEN_20261103.md) | Pinned publisher revision `a53b7b5784afdd50b2fda9195c9f724ef75ffdaf`, CC BY 4.0 according to publisher card. Human-video consent scope is unknown. | Continuous action/state trajectories; no verified scene ID or frozen supplied-option task requiring video and state. | No model-quality result. | A sampled part is 37.765 GiB; no payload was acquired. | **Review only. Do not acquire until grouping, consent scope, task, and resource gates are resolved.** |

## Current operational state

- Worktree head and `origin` were both `6e42213ed886f1d8e5d16b9df626eb6f8d618d58` at inspection; repository changes are limited to this note and its JSON record. Existing untracked PhysicalAI split manifests remain untouched.
- C: had about 1.40 GiB free, below the RoboMNIST extractor's fixed 2 GiB reserve. Quartus compile processes were live. They were not stopped or modified.
- Large artifact storage was reviewed in `R0_ARTIFACT_STORAGE_AUDIT_20261103.md`. It classifies source media, feature caches, checkpoints, and partial downloads as retained or pending evidence. No item was proven safely disposable. Prior automatic deletion rejection was not retried or routed around.
- No new training, inference, media access, model loading, or sealed-audit access occurred for this matrix.

## Next action

Resume the **original** RoboMNIST feature cache only when a stable capacity
sample meets its existing guard. Extract the two missing validation features,
verify all 1,780 receipts, then evaluate the already frozen train-only
checkpoint on all 2,000 validation rows once. Keep the previous train-only
checkpoint unchanged. This evaluation will test this narrow development task;
it cannot by itself establish general multimodal quality or close R2-M without
the predeclared unimodal and perturbation controls.

If capacity does not recover without deleting unverified reusable artifacts,
do not bypass the guard. Continue only with metadata/code review that does not
create another large artifact, and report the resource blocker.
