# R2: role-supervised supplied-option QA — executed 2026-10-14

## Outcome

Both fixed50-epoch arms actually completed650 updates each. All99 train videos/
198 questions and all257 observed-development videos/514 questions were retained.
Best/final checkpoints and their original epoch outputs replayed bitwise, including
a fresh process. Execution PASS; **QA benefit UNCERTAIN; no adoption or epoch extension**.
This is one Video+Text component comparison, not a four-modality95–98% result.

[Fixed protocol](R2_ROLE_SUPERVISED_QA_PROTOCOL_20261014.md),
[machine-readable results/hashes](R2_ROLE_SUPERVISED_QA_RESULTS_20261014.json),
[all100 learning-curve rows](R2_ROLE_SUPERVISED_QA_CURVES_20261014.csv).

## Design and source-label integrity

Reuse existing frozen256×512 TAPIR trajectory/initial-appearance tokens and1536-d
supplied-option text features; no new encoder extraction or encoder training.
Both arms share exactly the same139,521-parameter architecture, initial state,
seed17 sampler, AdamW0.001/weight_decay0.01, batch8videos and50epochs.
The only arm difference is auxiliary source-role loss weight0 versus1.
No source box/track ID selects inference points/crops/frames; all256 model tracks
remain. A shared small point/context encoder predicts cover membership and
candidate-conditioned attention uses its predicted logits. Original3-option CE
remains the main loss. Checkpoint rule: earliest minimum original-question dev NLL.
NLL is computed from target log-softmax, not floored softmax probabilities.

The initial standalone cover-only experiment failed before updates because one
train cover track begins at8, not0. Source inventory then found four dev clips
without the exact singleton grounded-cover label and14 flag/grounded discrepancies.
Flags were not substituted for missing grounded labels. New auxiliary coverage:
98train/253dev; original QA coverage remains99train/257dev. Valid boxes containing
no grid point are retained as all-negative membership examples. Dev grid support
is251/253 labeled clips. Supplemental membership measures do not replace the full
QA denominator or justify dropping examples. Missing-label IDs/reasons are in config.

Labels disable auxiliary loss only, never QA loss or QA evaluation:
| Split | Video | Missing auxiliary label |
|---|---|---|
| train | video_3785 | one frame-zero cover box required |
| valid | video_2109 | one grounded cover required |
| valid | video_2163 | one grounded cover required |
| valid | video_6174 | one grounded cover required |
| valid | video_6398 | one grounded cover required |

No data/config/checkpoint from legacy teachers or previous runs was overwritten.
Official script×participant grouping does not establish participant-disjointness.
These are previously observed development data used for checkpoint selection,
not a sealed/blind audit. Source cuts, including legacy cut=-1, are unchanged;
the previous source-viewpoint/reveal limitations remain. No seed/prompt sweep.

## Selected original QA results — all514 questions

| Arm | Best epoch/update | Accuracy | NLL | Brier | ECE15 | Both conditions correct |
|---|---|---:|---:|---:|---:|---:|
| qa_only | 11/143 | 0.389105 | 1.081731 | 0.656851 | 0.038557 | 0.206226 |
| role_supervised | 15/195 | 0.402724 | 1.073662 | 0.651652 | 0.029866 | 0.217899 |

Role-minus-QA-only paired differences,10,000 video-ID resamples, seed17,
percentile95% intervals. Two questions from a video remain paired. These intervals
do not remove selection bias or guarantee actor/source independence.

| Metric | Difference | 95% interval |
|---|---:|---|
| accuracy | +0.013619 | [-0.023346, +0.050584] |
| nll | -0.008069 | [-0.018746, +0.002674] |
| brier | -0.005199 | [-0.012503, +0.002169] |

All intervals include zero. Accuracy gain is7/514 questions(+1.362pp),
95% interval[-2.335,+5.058]pp. NLL/Brier point estimates improve slightly,
but this does not establish a reliable QA improvement. No ECE uncertainty interval
was calculated. Do not promote the selected checkpoint as an Omni Teacher.

## Fixed selected-checkpoint controls and condition following

| Arm/control | Accuracy | NLL | Brier | ECE15 |
|---|---:|---:|---:|---:|
| qa_only/normal | 0.389105 | 1.081731 | 0.656851 | 0.038557 |
| qa_only/zero | 0.389105 | 1.089689 | 0.660716 | 0.026162 |
| qa_only/mismatch | 0.389105 | 1.090932 | 0.662732 | 0.059831 |
| qa_only/temporal_zero | 0.389105 | 1.090603 | 0.661274 | 0.030662 |
| role_supervised/normal | 0.402724 | 1.073662 | 0.651652 | 0.029866 |
| role_supervised/zero | 0.389105 | 1.089303 | 0.660702 | 0.011087 |
| role_supervised/mismatch | 0.365759 | 1.097635 | 0.666827 | 0.044668 |
| role_supervised/temporal_zero | 0.422179 | 1.073602 | 0.650066 | 0.016838 |

QA-only predicts the second canonical option on all514 questions: zero, mismatch
and temporal-zero state controls change no argmax predictions. The role-supervised
arm changes85/143/105 predictions under these controls, but dependence is not
correct reasoning. Zeroing temporal columns4:128 raises its Accuracy to42.218%
and slightly lowers NLL; **useful temporal contribution is not demonstrated**.
Role-supervised initial/final predictions differ on only5/257 videos.
All127 changed-target condition pairs have0 both-correct in BOTH selected arms.
Retain this failure of conditional use despite the small aggregate point gain.

Supplemental cover top1 point-in-source-bbox improves1/253→55/253
(0.395%→21.739%). This is partial role localization, not hidden-item binding
or QA Accuracy. Natural-grid membership metrics are background-dominated:
positive fraction3.239%. They cannot satisfy the95–98% Omni goal.
Supplemental binary NLL uses1e-12 probability clipping; it never selects QA weights.

## Learning curves and overfitting

| Arm/epoch | Train Acc | Dev Acc | Train NLL | Dev NLL |
|---|---:|---:|---:|---:|
| qa_only/1 | 0.449495 | 0.389105 | 1.068489 | 1.088584 |
| qa_only/5 | 0.449495 | 0.389105 | 1.061105 | 1.085760 |
| qa_only/11 | 0.449495 | 0.389105 | 1.045450 | 1.081731 |
| qa_only/15 | 0.570707 | 0.365759 | 0.948891 | 1.121767 |
| qa_only/20 | 0.722222 | 0.391051 | 0.616397 | 1.544985 |
| qa_only/30 | 0.747475 | 0.369650 | 0.416106 | 1.955927 |
| qa_only/40 | 0.747475 | 0.357977 | 0.391855 | 2.155503 |
| qa_only/50 | 0.838384 | 0.365759 | 0.312611 | 2.329872 |
| role_supervised/1 | 0.449495 | 0.389105 | 1.074268 | 1.089839 |
| role_supervised/5 | 0.449495 | 0.389105 | 1.061380 | 1.087541 |
| role_supervised/11 | 0.449495 | 0.389105 | 1.050860 | 1.083668 |
| role_supervised/15 | 0.510101 | 0.402724 | 1.023355 | 1.073662 |
| role_supervised/20 | 0.676768 | 0.379377 | 0.799856 | 1.137737 |
| role_supervised/30 | 0.742424 | 0.350195 | 0.450424 | 1.814469 |
| role_supervised/40 | 0.747475 | 0.338521 | 0.397924 | 2.134531 |
| role_supervised/50 | 0.782828 | 0.359922 | 0.371770 | 2.456419 |

Final train Accuracy/NLL is83.84%/0.312611 versus dev36.58%/2.329872
for QA-only, and78.28%/0.371770 versus35.99%/2.456419 for role supervision.
Both clearly overfit after their early selected checkpoints. Training metrics
here are full fixed train-corpus predictions, not rolling minibatch estimates.
More repetition of this same99-video corpus is not supported as the next step.

## Training amount and artifacts

| Arm | Run videos/questions | Selected videos/questions | Unique train videos |
|---|---|---|---:|
| qa_only | 4950/9900 | 1089/2178 | 99 |
| role_supervised | 4950/9900 | 1485/2970 | 99 |

4,851 repeated video presentations per arm after first exposure; two questions
per video are distinct conditions, not independent media. Best and final weights,
optimizer/sampler/Torch RNG states, ordered epoch samplers, all100 epoch NPZs,
log probabilities, histories and controls are local under
`C:/CodexArtifacts/pretrained-reuse-r2/role-supervised-qa-full-01/`.
Strict resume was not implemented/tested; saved states do not imply a resume claim.

116 run files were independently hash-checked; all100 epochs had full primary
denominators and recomputed stable NLL. Both arms' best AND final probabilities,
cover probabilities and QA log probabilities replayed bitwise against original
saved train/dev epoch outputs in a fresh process. No tolerance relaxation.

Source commit `46820d9ad10215677d2aec92474d1e08f62709cf`.
Exact command: `python -m scripts.train_role_supervised_qa --output
C:/CodexArtifacts/pretrained-reuse-r2/role-supervised-qa-full-01`.
Config SHA `eb80e86d0d0c732ee67e0879f6bb7dc3e7a36755e2dda1093f0c6a7ce8d72715`;
result `7a2ae393d17a4080d92e721d2040637fde1230de0ebdf794767bac0d4e6681cf`.
Frozen [TAPIR](R1_TAPIR_ADMISSION_20261013.md) source/weights and MiniLM
revision4ca70771034acceecb2e72475f72050fcdde4ddc remain pinned; actual weight
hashes rechecked unchanged. Source/pair/media/feature/code hashes in the config
and publication ledger preserve provenance. No weights/media/NPZ enter Git.

## Cost, verification and failed attempts

Intel i7-12700H, Python3.11.9/Torch2.6.0+cu124, FP32 CPU4/deterministic.
Fresh frozen features were NOT extracted for this experiment; inherited extraction
cost remains in the prior TAPIR report. Main body88.675800s; arm bodies35.241732s
and35.435703s include their evaluation/checkpoint/control work. Separate fresh
reload18.947139s. These scopes exclude import/driver startup and are not complete
raw-video Decision latency. PID70740 started09:34:38.415JST and exited0; exit was
observed at00:36:34UTC, not an exact end timestamp. Live observed process peak
working set2,058,072,064bytes; final process peak/total CPU time unmeasured.
GPU training allocation/time0, local existing hardware, cloud cost0.

Frozen video31,072,327 + text17,389,824 + readout139,521 =48,601,672 parameters.
Raw frozen weight files193,917,962bytes + selected readout561,045bytes =
194,479,007bytes before tokenizer/config/code/runtime. This is not a built
<=200MB package or full Omni resident accounting; Audio/Image/Sensor paths are
not part of this component cost. Quality/package/runtime gates remain open.

Ruff passed;995 CPU tests passed and1 skipped in15.15s. Model/dataset manifests,
four catalogs and example policy audit passed. Fresh-context review caught NLL
flooring plus replay/helper-provenance gaps; RED→GREEN regressions and re-review
resolved all findings before the committed-source full run.

All attempts are preserved: standalone missing-box preflight(no updates); two
aborted schema inventories before complete inventory03; three successful two-update
train-only smokes; complete full comparison; auditor01 metadata st_size call error
(no numeric result) corrected by auditor02; direct fresh-replay import failed
before model load, then unchanged script passed with repository import context.
Optional bundled Matplotlib was unavailable; no install occurred. CSV provides
all curves; no plot was generated. Failed attempts never selected model weights.

## Next useful experiment

Close this fixed auxiliary-loss comparison without promotion or another seed/epoch
sweep. Role supervision improves partial localization but does not establish
hidden-object binding, actor-relative option meaning or changed-condition use.
First-frame appearance still omits later-revealed objects; sparse temporal spatial
features were [actually preserved on eight train clips](R2_TEMPORAL_APPEARANCE_20261014.md).
The next representation comparison should preserve later appearance and spatial
relations, with the same original QA corpus/options and input-only target handling.
Data scarcity, annotations/viewpoints and readout/encoder limitations remain
possible contributors; this result does not prove a model-family ceiling.
Text/Image/Audio/Video quality, useful multi-input gain, OOD, calibrated JSON and
the final lightweight runtime remain required; the overall goal is incomplete.
