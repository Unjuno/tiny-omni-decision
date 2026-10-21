# Perception Test AV label-strata audit (2026-10-21)

## Scope

This is a CPU-only, post-hoc annotation audit of the already inspected AV1
development cohort. It joins saved audio-only, video-only and joint predictions
to the pinned source label after inference. The event label was not a model
input and was not used to fit, select, filter or calibrate a model. No media was
decoded, no checkpoint was loaded, and no train, validation or sealed-audit
model evaluation was run. This does not establish a product result or a blind
test result.

The cohort has 256 validation rows, one event per video, with a balanced target
by construction. It is therefore not representative of the official validation
prior (91.33% visible). The audit verifies sample/event identity and target
agreement for every saved prediction before calculating category summaries.

## Reproducible inputs

- Source revision: Perception Test `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- Cohort: `fixed_av_cohort_v1_seed17.json`, SHA-256
  `dd8c9de6d8970eb1105cb81551e386fca563eb174b8bcda5b09dcc8a16e567be`.
- Train `sound_localisation` annotation archive, SHA-256
  `0200a50644ab9d36a2dd80e5239e709cabeeaf2e4274eca201c90743b89cc6f5`.
- Official validation `sound_localisation` annotation archive, SHA-256
  `4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141`.
- Matched-readout result, SHA-256
  `95ee830089d9211f2b7d68929eda2020e02be958b77331ebd18c29636986a224`.
- Analysis report: `C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\posthoc-category-error-audit-02\report.json`, SHA-256
  `55d44338460156a21af3f1a323b2ad3ece20eb79343a6de7bed532d5d055baba`.
- Command (from the repository root):

  ```powershell
  python scripts/audit_perception_av_error_strata.py `
    --cohort C:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-screen-20261021\fixed_av_cohort_v1_seed17.json `
    --annotations C:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-screen-20261021\sound_valid_annotations `
    --train-annotations C:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-screen-20261021\sound_train_annotations `
    --readout-dir C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\matched_linear_readout_v1 `
    --output C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\posthoc-category-error-audit-02\report.json
  ```

## Findings

In the balanced cohort, event categories were not balanced across visibility:

| Source label | n | visible | not visible | audio accuracy | video accuracy | joint accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Interaction:hitting | 92 | 82 | 10 | 50.0% | 67.4% | 63.0% |
| Animal:sound | 61 | 0 | 61 | 52.5% | 41.0% | 59.0% |
| Human:Speech (source label has leading whitespace) | 42 | 1 | 41 | 73.8% | 52.4% | 78.6% |
| Object:sound | 34 | 25 | 9 | 55.9% | 58.8% | 70.6% |
| Interaction:dropping | 12 | 7 | 5 | 50.0% | 83.3% | 58.3% |

Remaining categories have at most seven examples each; their counts and
metrics are retained in the machine-readable report. The source-category
mixture is a plausible confound for this small balanced selection. This
post-hoc analysis cannot determine whether any readout used category-correlated
features, so it neither proves nor disproves semantic audio-video fusion.

The pinned full source annotation sets also show strong category/label
associations. Counting videos as groups, rather than treating segments as
independent, yields:

| Source label | Train segments: visible / not | Train videos: visible / not / both in same video | Official validation segments: visible / not | Validation videos: visible / not / both in same video |
|---|---:|---:|---:|---:|
| Interaction:hitting | 7,206 / 49 | 1,705 / 29 / 23 | 18,972 / 132 | 4,523 / 81 / 63 |
| Object:sound | 2,209 / 75 | 1,006 / 55 / 34 | 6,081 / 134 | 2,733 / 99 / 51 |
| `Human:Speech` | 114 / 484 | 58 / 312 / 15 | 366 / 1,259 | 173 / 849 / 48 |
| Animal:sound | 4 / 583 | 4 / 340 / 2 | 13 / 1,487 | 12 / 879 / 7 |
| Interaction:rolling | 566 / 1 | 232 / 1 / 0 | 1,451 / 6 | 574 / 4 / 2 |
| Human:Sound(clapping) | 452 / 3 | 135 / 3 / 0 | 1,225 / 3 | 353 / 3 / 0 |

Across all source labels, 240 train videos and 190 official-validation videos
contain both visibility states somewhere among their sound segments. These are
not necessarily matched or independently adjudicated counterfactuals. Only a
minority of categories provide substantial same-video support; most observed
class variation is between videos. The audio/video task therefore needs a
stronger, predeclared conditional design before it can support a both-inputs-
required claim.

The earlier 256-example matched readout remains unchanged: accuracy was 56.25%
audio-only, 57.81% video-only, and 64.84% joint. Its 2,000-replicate paired
video bootstrap showed the joint-minus-audio Accuracy interval above zero,
while NLL/Brier intervals crossed zero. Readout training fit 100% of its 256
training examples. Those results are still narrow, previously observed
development evidence; this stratification neither changes nor selects a
model.

## Decision

Keep Perception Test visibility as a low-quality exploratory AV diagnostic.
Do not use this cohort to tune the model or claim general multimodal
complementarity. Any follow-up must preregister a video-grouped task and
conditional label support, compare audio/video/joint on identical inputs and
options, and use a fresh independent validation cohort. This audit does not
open the official test or any sealed audit.
