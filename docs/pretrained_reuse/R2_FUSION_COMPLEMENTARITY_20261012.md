# VIDIMU saved-prediction complementarity audit

## Outcome and next action

The current 39-episode / three-subject development set cannot demonstrate an
Accuracy gain from adding video to the selected sensor model: sensor already
answers every episode correctly. Stop additional fusion fitting on these IDs.
This is a limitation of this measurement task and selected models, not a claim
that video is unnecessary generally or that the model family cannot improve.

The next joint-quality experiment needs a predeclared task whose target requires
complementary observations. Before training, establish source rights, natural
paired observations, grouped splits, timestamp/availability semantics and enough
independent validation groups. Do not manufacture a gain by deleting the sensor's
correct examples, corrupting it after seeing outcomes, or averaging unrelated
taxonomy probabilities. Missingness, if studied, must be a separate predeclared
robustness condition; clean input performance must remain reported.

## Measured diagnostic

Read existing selected-checkpoint validation predictions only. Exact ordered
subject/activity/trial IDs and targets must agree; duplicate IDs are rejected.
No model inference, training, calibration, seed search or audit access occurred.

| Observation | Episodes |
|---|---:|
| Both sensor and video correct | 26 |
| Sensor correct, video wrong | 13 |
| Video correct, sensor wrong | 0 |
| Both wrong | 0 |
| Joint correct | 30 |
| Sensor correct but joint wrong | 9 |
| Both unimodal models correct but joint wrong | 2 |

Joint-minus-sensor NLL is +0.408505 overall and positive for every subject:
S42 +0.632862, S47 +0.258124, S57 +0.334529. This comparison uses the original
selected sensor/video/linear-joint arms, not whichever later fusion arm looks
worst. Prior blockscale and gated results remain separately recorded.

Video assigns higher true-class probability than sensor in 12 cases. A
**label-aware selection oracle** taking the lower of their losses gives NLL
0.143734 versus sensor 0.163704 (difference 0.019970). This is an optimistic
diagnostic of selecting between these two fixed predictions, not a deployable
selector, not a general fusion lower bound, and not authorization to tune a
selector on these labels. Its Accuracy remains 100%, with no additional correct
cases. Joint NLL is 0.572209; joint has lower loss than sensor in only 11 cases.

Repeated validation use, checkpoint selection and only three independent
subjects preclude broad generalization or blind-result claims. No confidence
interval is newly inferred from 39 supposedly independent episodes.

## Reproduction

`python -m scripts.diagnose_vidimu_complementarity --output C:/CodexArtifacts/pretrained-reuse-r2/vidimu-complementarity-01`

The fresh output contains all input SHA-256 hashes, overall and subject-level
statistics. Source files were checked unchanged after analysis. Four CPU unit
tests cover rescue/harm accounting and invalid probability rejection. No weights
or media are added to Git. This diagnosis changes the next experiment selection;
it does not satisfy the product's joint 95–98% gate.
