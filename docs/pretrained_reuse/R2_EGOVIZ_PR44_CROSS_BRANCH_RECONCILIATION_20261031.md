# R2-M cross-branch reconciliation: EgoViz PR #44 — 2026-10-31

## Provenance and scope

This records an already completed fixed-checkpoint evaluation from open Draft
PR #44, branch `codex/r2-egoviz-fresh-session-validation-20261030`, head
`a1d86d6f30797953b2e07ff096263176910378c6`. The protocol and detailed report
are in that branch:

- `R2_EGOVIZ_FRESH_GROUP_GENERALIZATION_PROTOCOL_20261030.md`
- `R2_EGOVIZ_FRESH_GROUP_GENERALIZATION_RESULTS_20261030.md`

No media, features, model weights or predictions were reread or recomputed for
this reconciliation. No training, checkpoint selection, tuning, GPU work,
sealed-media access or product promotion occurred here.

## Existing result

The frozen VideoPrism-B + MOMENT readouts were evaluated on 2,394 four-choice
video-to-IMU timestamp-correspondence decisions from 69 session groups absent
from the prior training and repeatedly viewed development corpora. Reported
Accuracy / NLL / Brier / ECE were:

| Fixed checkpoint | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Linear, historically selected step 616 | 55.22% | 1.1981 | 0.6508 | 0.2105 | 0.3417 |
| Linear, fixed step 5,632 | 53.26% | 1.2254 | 0.6655 | 0.1984 | 0.3342 |
| Ordered cross-modal GRU, fixed step 5,632 | 57.10% | 4.7958 | 0.7865 | 0.3783 | 0.9492 |

The paired 69-session bootstrap for GRU minus linear endpoint reported
Accuracy `+3.84 pp` (95% interval `+1.66` to `+6.22`), NLL `+3.5703`
(`+3.1068` to `+4.0567`), Brier `+0.1210` (`+0.0655` to `+0.1772`), and ECE
`+0.1798` (`+0.1318` to `+0.2321`). Here positive NLL/Brier/ECE differences
are worse. The GRU's Accuracy gain is paired with severe overconfidence and
does not establish better probability quality.

Controls reported in the earlier 44-session development result are not new
independent outcomes on these 69 groups. The fresh-set report includes no
sensor-only, video-only, metadata-only or uniform baseline, so matched
unimodal/metadata controls remain open for this fresh cohort. The 28.66%
sensor-only score in PR #38 belongs to the earlier 44-session development
evaluation and must not be compared as if it came from these 69 sessions.

## Gate interpretation

This is useful, narrow evidence that a time-ordered video+sensor scorer can
transfer above the sensor-only control on this constructed correspondence
task. The target is derived from source timestamps, not an adjudicated natural
action decision. The pinned dataset card defines `group_id` as a session but
does not establish that workers are unique across different groups; participant
independence remains unknown. Scores remain far below the per-task 95% target,
and the GRU probabilities are poorly calibrated.

Treat this as diagnostic evidence for the video+sensor branch of R2-M only. It
does not close M-2's broader modality/task matrix, establish product quality,
or justify a new fit or tuning pass on these now-observed groups. Keep the
existing artifacts and PR #44 unchanged. The next model-quality evaluation
requires a separately admitted task/cohort with trusted task labels,
participant-level provenance, fixed unimodal/metadata controls and an
independent development design.
