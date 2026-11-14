# R2 multimodal task admission matrix — 2026-11-04

## Purpose and original scope

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
| [RoboMNIST AV repetition match](R2_ROBOMNIST_AV_MATCH_PREFLIGHT_20261103.md) | CC BY 4.0, official Figshare v1; source/article pins in `R2_ROBOMNIST_SOURCE_20261014.md` and `R2_ROBOMNIST_EPISODE_TASK_FEASIBILITY_20261103.md` | Frozen whole-repetition AV match; 780 train / 1,000 validation groups, pair-unit split. Episode-level AV synchronization is supported; frame-level alignment is not. | Follow-up: joint accuracy 0.5335 vs 0.5000 unimodal, paired 95% interval +0.0130 to +0.0550; joint NLL/Brier/ECE-15 1.2494/0.7217/0.3074. | Cache completed 1,780/1,780. Validation is previously exposed development data, not blind or scene-held-out. Cross-task metadata overlap is documented. | **Do not retune or promote: modest task-specific accuracy signal with severe calibration failure.** |
| [DCASE 2025 / STARSS23 AV match](R2_DCASE2025_AV_MATCH_CV_20261101.md) | STARSS23 v1.1 official MIT release; bounded source audit at `R2_STARSS23_SOURCE_20261011.md` | Four-choice recording identity match; grouped by 78 parent recordings and 9 rooms. | Nested-room CV: joint 26.56% vs 25% chance; NLL/Brier worse than uniform; paired intervals include zero. Direct cosine: 27.34%, uncertainty wide. | Existing frozen development features; public dev-test and sealed audit not used in these reports. Source sample is small and room clusters are few. | **Do not tune or repeat. Weak/negative evidence; does not pass both-input quality gate.** |
| [Eidon Tracker POV + IMU](R2_EIDON_TRACKER_POV_MATCH_20261029.md) | Pinned video and IMU revisions; cards report CC BY 4.0. Rights do not remove privacy concerns in unblurred private-home footage. | Video-to-sensor recording match; contributor/task grouping. | Pooled and LSTM: 7/24 (29.2%) on 3 held-out contributors; reversed-time LSTM was slightly better. Strong overfit; no useful temporal or both-input effect. | Tiny, already exposed development set. | **Do not repeat or promote. Insufficient independent groups and negative control evidence.** |
| [NVIDIA PhysicalAI SingleArm](R2_PHYSICALAI_FRESH_ACTION_PREFLIGHT_20261103.md) | Publisher revision `73c00c18e7ca108e91846294b20a1d7cd5de9340`, CC BY 4.0 per tracked source screen. | Video + proprio action-prototype prediction; existing episode split. No scene/layout identity, and review found shared simulated workspace. | Existing pilot: joint 72.5% vs proprio 70.4%, but episode-bootstrap interval includes zero; heads reached 100% train accuracy and calibration is mixed. This is not robust both-input benefit. | Fresh split metadata exists in tracked report, but acquisition attempt stopped at HTTP 429 with 175 files; current free space is below reserve. The 1,000 reserved IDs remain unopened. Do not read user-owned untracked manifests. | **Not admitted. Scene independence and reliable complementarity remain unproven; no acquisition or refit now.** |
| [HiFi-UMI-2K](R2_HIFI_UMI_2K_SOURCE_SCREEN_20261103.md) | Pinned publisher revision `a53b7b5784afdd50b2fda9195c9f724ef75ffdaf`, CC BY 4.0 according to publisher card. Human-video consent scope is unknown. | Continuous action/state trajectories; no verified scene ID or frozen supplied-option task requiring video and state. | No model-quality result. | A sampled part is 37.765 GiB; no payload was acquired. | **Review only. Do not acquire until grouping, consent scope, task, and resource gates are resolved.** |

## Post-evaluation update — 2026-11-04 11:27 JST

The RoboMNIST candidate's fixed cache was completed and its frozen train-only checkpoint was evaluated once on the full 2,000-row development validation set. Joint AV accuracy was 0.5335 versus 0.5000 for each unimodal control; the paired 95% interval was +0.0130 to +0.0550 over 500 pair units. Joint calibration was poor (NLL 1.2494, Brier 0.7217, ECE-15 0.3074), so this is a narrow accuracy signal, not a passing quality result. The validation is already exposed and must not be used for further tuning. Details and hashes are in [the evaluation record](R2_ROBOMNIST_AV_MATCH_EVALUATION_20261104.md).

At the follow-up, C: had about 7.8 GiB free; the extractor's 2 GiB guard was retained. Quartus PID 1732 remained active and untouched; the RTX 3080 was idle. The original cache is now `COMPLETE` at 1,780/1,780. No media, checkpoint or cache was deleted, overwritten, or added to Git, and sealed audit remained unopened. The two user-owned untracked PhysicalAI manifests were not read or modified.

## Current operational state at the original metadata-only snapshot

- Worktree head and `origin` were both `6e42213ed886f1d8e5d16b9df626eb6f8d618d58` at inspection; repository changes are limited to this note and its JSON record. Existing untracked PhysicalAI split manifests remain untouched.
- C: had about 1.40 GiB free, below the RoboMNIST extractor's fixed 2 GiB reserve. Quartus compile processes were live. They were not stopped or modified.
- Large artifact storage was reviewed in `R0_ARTIFACT_STORAGE_AUDIT_20261103.md`. It classifies source media, feature caches, checkpoints, and partial downloads as retained or pending evidence. No item was proven safely disposable. Prior automatic deletion rejection was not retried or routed around.
- No new training, inference, media access, model loading, or sealed-audit access occurred for this matrix.

## Next action

Do not retune or re-evaluate RoboMNIST validation. Screen a genuinely fresh,
rights-cleared, independently grouped task for modality-native and joint Path A
evaluation. Before any fit, require sufficient independent groups, verified
labels, task necessity for both streams, and a clean train/calibration/dev split.
The prior matrix's DCASE and Eidon candidates remain weak/negative; PhysicalAI
scene identity remains unresolved, and user-owned split manifests are excluded
from this review. If no candidate passes these gates, continue R0/R1 feasibility
work instead of fitting another weakly identified task.

## Residual RoboMNIST cohort follow-up — 2026-11-04

A read-only split-code/manifest screen found only 10 groups unselected from the prior non-sealed eligible pool of 1,790 (1,780 already assigned to train/validation). The other 345 of 2,135 total source groups were outside that pool and were not inspected. Ten groups are far below the task's provisional 1,000-group validation gate. Do not train or re-evaluate this task; see [the residual-group screen](R2_ROBOMNIST_RESIDUAL_GROUP_SCREEN_20261104.md). This does not establish a clean fresh evaluation cohort.

## CREMA-D source-code follow-up — 2026-11-04

Pinned source code confirms same-filename `VoiceVote`, `FaceVote`, and `MultiModalVote` columns, so a common AV target is structurally possible. The repository requests usage registration, while individual voice/face rights, consent scope, and a verified actor/sentence split remain unresolved. Keep this narrow emotion source `REVIEW_NOT_ADMITTED`; no data access, registration, or fit. See [source-code follow-up](R2_CREMA_D_AV_SOURCE_SCREEN_20261102.md#pinned-source-code-follow-up--2026-11-04).

## 2026-11-10 Glasgow AV-radar speech source screen

The University of Glasgow repository labels the dataset CC BY 4.0, but
identifiable audio/video is controlled-access and the reported 800 samples
come from only four speakers. Its approximately 17 GB archive and four
independent speaker groups do not support disjoint train, calibration, and
development evaluation; necessity of all AV-radar inputs is also unproven.
No payload or access request was made. Keep `REVIEW_NOT_ADMITTED`; see [the
source screen](R2_GLASGOW_AV_RADAR_SPEECH_SOURCE_SCREEN_20261110.md).
