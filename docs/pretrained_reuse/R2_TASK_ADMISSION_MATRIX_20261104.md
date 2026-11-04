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

## 2026-11-05 RoboMNIST task-native CV follow-up

The frozen TRAIN-only five-fold condition CV is complete for the separate
10-option “which digit did the robot draw?” task. It used 780 TRAIN source
groups across 60 arm × speed × digit conditions; complete conditions were held
out together. Original validation arrays and sealed audit stayed closed.
At step 25, joint AV reached 20.64% accuracy (10% chance), versus 18.72%
audio-only and 10.90% video-only, but joint NLL/ECE were 12.929/0.701.
By step 400, all readouts fit their training folds while OOF calibration and
class recall remained poor. This is condition-level overfitting, not a useful
decision-quality pass and not evidence that the target needs both modalities.
Do not tune or select another checkpoint on these OOF folds or the already
exposed original validation.

The matched ordered-vs-shuffled recurrent follow-up remained near chance.
At 50 updates ordered accuracy/NLL/ECE were 50.0%/0.706/0.020, versus
49.3%/1.370/0.337 shuffled and 50.2%/0.716/0.083 pooled. At 400 updates,
ordered accuracy fell to 48.9% and NLL rose to 0.794; it did not beat pooled
accuracy/NLL (53.2%/1.245). This does not justify a larger LSTM/xLSTM or more
RoboMNIST fitting. Preserve both studies as negative, task-scoped development
evidence; they do not establish a general architecture limit.

Full provenance and curves: [10-option Decision results](R2_ROBOMNIST_DIGIT_DECISION_RESULTS_20261105.md),
[temporal readout results](R2_ROBOMNIST_TEMPORAL_CV_RESULTS_20261105.md).
The candidate remains **not admitted for product promotion**. Next fitting work
still requires a distinct source with explicit rights, fresh independent
groups, and a frozen task/control design; do not substitute another model
variant on the same exposed RoboMNIST source.

## 2026-11-05 Action Verb Corpus source screen

The OFAI resource page presents a small synchronized audio/video/motion task
and says CC BY 4.0, but its linked Zenodo release labels the dataset
CC BY-NC-SA 4.0. Audio also contains spoken descriptions of the action, so
both-input necessity is not established, and only 12 participants/46 episodes
are available. Keep it REVIEW_NOT_ADMITTED; no payload was accessed. See
the [source screen](R2_ACTION_VERB_CORPUS_SOURCE_SCREEN_20261105.md).
