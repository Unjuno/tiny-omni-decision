# R2-M supplied-activity Decision comparison (2026-11-21)

## Decision

A fresh 21-choice activity readout can classify paired audio and vibration, but
the gated audio+sensor arm did not consistently improve on audio alone. It had
lower validation top-1 and macro Accuracy, slightly better macro NLL and ECE,
and slightly worse macro Brier. The selected sensor-only arm was substantially
weaker. Keep the gated block as a research candidate; do not adopt it as a
product model or claim general multimodal competence.

This is a closed-set activity-label task: the choices are the 21 activity IDs
listed in the effective config, and each choice is represented to the readout
by a one-hot query. It does not consume natural-language option descriptions,
video, or general text/image inputs. It is evidence about one audio+vibration
task only, not the four-modality 95–98% objective.

## Integrity, data and task

- Source: Zenodo `10.5281/zenodo.7937591`, admitted for local development under
  CC BY 4.0; only the already verified `multisensor-1` frozen feature cache was
  used. No raw audio/vibration was copied into Git.
- Frozen participant split v2 SHA-256:
  `0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0`.
  Train participants: 2, 4, 7, 8, 10, 13, 14; calibration: 12, 17;
  validation: 9, 15. Participant groups do not overlap.
- Pair-index metadata SHA-256:
  `faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0`;
  internal pair-index digest:
  `d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276`.
- Frozen feature-cache index SHA-256:
  `61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498`.
  It contains 6,971 unique trial assets: 5,449 train, 760 calibration, 762
  validation. The allowlisted metadata has 7,018 rows; 47 allowed trials were
  not part of the pair-index-derived feature cache and were not used.
- Of 24 source activities, only 21 appear in all three development splits.
  `11_typing`, `14_sweep`, and `15_vacuum-cleaning` have no validation examples
  (and no calibration examples), so they were excluded by a coverage rule fixed
  before model scoring. This is a 21-activity benchmark, not coverage of all 24.
- Used examples after this coverage rule: train 5,424, calibration 760,
  validation 762. Uniform 21-choice chance macro Accuracy is 4.76%; validation
  sample-weighted uniform Accuracy is 7.61% because class counts are uneven.
- Sealed-audit signals were not loaded. The validation set is development data
  and has only two participant clusters; no blind-audit claim is made.

## Pipeline and attempts

All three arms use the same frozen EfficientAT DyMN-0.4 audio features, fixed
split, 21 activity choices, seed 17, batch size 32, 80 epochs, AdamW
(`lr=1e-3`, weight decay `0.01`), cosine schedule, train-only inverse-frequency
class weighting, and random candidate order per training batch. Each epoch
visits every eligible train sample once. Calibration macro NLL selects the
checkpoint; training always runs all 80 epochs. The base audio encoder and all
features remain frozen.

**Attempt 1 — incomplete, preserved.** The first fresh-readout run completed
audio-only at 80 epochs, then safely stopped during sensor-only epoch 23 after
`clip_grad_norm_` returned a non-finite gradient norm. The largest previously
recorded finite sensor-arm norm was about `3.16e15`; the fused arm did not start.
This is an execution safety stop, not a quality failure. The cached vibration
channels have strongly different offsets and scales. Follow-up used only
supported train examples to estimate per-axis means `[-10516.045, -7948.597,
34.491]` and standard deviations `[28.580, 35.918, 22.607]`. The association
with the gradient instability is plausible, not a proof of sole causation.
Attempt-1 checkpoints, histories, source hashes and stop record remain in its
own output directory.

**Attempt 2 — completed.** The only intended modeling change was train-only,
per-axis z-score normalization of the three vibration channels, applied with
the same train-derived statistics to calibration and validation. The frozen
feature cache was not rewritten. All arms started from fresh readout weights;
no attempt-1 weights were reused. Each arm completed 13,600 updates over 80
epochs, consuming 433,920 train example presentations and 5,424 unique train
examples. The 80-epoch final checkpoint is separate from the calibration-best
checkpoint.

| Arm | Trainable params | Selected epoch | Train macro Acc / NLL | Validation Acc | Validation macro Acc | Macro NLL | Macro Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio only | 133,538 | 5 | 93.92% / 0.172 | 75.07% | 66.83% | 0.9785 | 0.4664 | 0.1229 |
| Sensor only | 96,962 | 4 | 39.42% / 1.664 | 27.56% | 30.11% | 2.1305 | 0.8001 | 0.1030 |
| Audio + sensor gated | 180,387 | 5 | 89.40% / 0.303 | 71.92% | 64.80% | **0.9455** | 0.4698 | **0.0643** |

Selected validation overall NLL/Brier were 0.8058/0.3677 for audio-only,
2.3224/0.8428 for sensor-only, and 0.8207/0.3923 for gated fusion. The fused
model's improved macro NLL and ECE do not offset its lower sample-weighted and
macro Accuracy; Brier also worsened. The training-to-validation macro-Accuracy
gaps were 27.09, 9.31, and 24.60 percentage points respectively.

### Learning curve and overfitting

Calibration is checked each epoch, but the table below samples fixed points;
validation is not used for checkpoint selection. Train Accuracy is the
within-epoch rolling metric, not a held-out generalization estimate.

| Arm / epoch | Train Accuracy | Calibration macro Accuracy | Calibration macro NLL |
|---|---:|---:|---:|
| Audio / 1 | 50.90% | 68.81% | 1.0322 |
| Audio / 5 (selected) | 90.89% | 78.56% | **0.8501** |
| Audio / 20 | 97.66% | 81.63% | 1.2844 |
| Audio / 40 | 99.67% | 79.56% | 2.7950 |
| Audio / 80 | 100.00% | 79.13% | 3.5733 |
| Sensor / 1 | 15.21% | 24.58% | 2.3672 |
| Sensor / 4 (selected) | 36.97% | 29.06% | **2.2975** |
| Sensor / 20 | 53.54% | 19.66% | 3.2029 |
| Sensor / 40 | 71.40% | 20.21% | 4.7415 |
| Sensor / 80 | 96.24% | 19.78% | 9.3491 |
| Gated / 1 | 36.26% | 52.02% | 1.4796 |
| Gated / 5 (selected) | 88.83% | 74.58% | **0.8482** |
| Gated / 20 | 96.74% | 77.42% | 1.5930 |
| Gated / 40 | 99.24% | 77.61% | 2.6438 |
| Gated / 80 | 100.00% | 77.76% | 3.6980 |

The fixed budget clearly overfits all arms by NLL. Final-epoch validation macro
Accuracy is audio 70.21%, sensor 28.73%, and fusion 71.62%, but macro NLL rises
to 5.245, 6.675, and 4.151; confidence averages rise to 0.979, 0.855, and
0.977. The calibration-selected checkpoints are retained and are not replaced
by these final weights.

## Fusion controls and uncertainty

For the selected fused checkpoint, validation macro Accuracy / macro NLL were:

| Condition | Macro Accuracy | Macro NLL | Macro Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Both inputs | 64.80% | 0.9455 | 0.4698 | 0.0643 |
| Audio absent | 12.07% | 5.2188 | 1.0731 | 0.2468 |
| Sensor absent | 47.91% | 2.0326 | 0.7363 | 0.2241 |
| Sensor shifted by one validation row | 64.14% | 0.9376 | 0.4661 | 0.0642 |

Removing either modality damages the gated model, but one-row sensor shuffling
barely changes its scores and slightly improves NLL. This does not show robust
sensor-content dependence or a net sensor benefit over the audio-only model;
audio-only validation macro Accuracy is 66.83%.

Paired sample comparison, fused minus audio-only, used 10,000 percentile
bootstrap replicates resampling unique trials within participant × target
activity strata (seed 17). Weighted Accuracy delta was −3.15 points (95% trial
bootstrap interval −5.64 to −0.66); macro Accuracy −2.01 points (−4.82 to
0.73); macro NLL −0.032 (−0.099 to 0.031); macro Brier +0.0036 (−0.0241 to
0.0307). These intervals are conditional on the two held-out participants,
not population-level subject uncertainty. By participant, audio-only vs fused
Accuracy was 63.83% vs 64.36% for participant 9 and 86.01% vs 79.27% for
participant 15. The effect is not consistent across these two people.

Exact reload of every selected checkpoint on the original CUDA runtime
reproduced the saved validation probabilities with maximum absolute delta
`0.0`. Candidate-order permutation changed fused probabilities by only
`3.58e-7`. A CPU reload attempt failed the then-used `1e-6` saved-probability
threshold; no numeric delta was captured, so CPU/CUDA parity is UNKNOWN. A
separate config-hash check also exposed Windows CRLF translation: canonical
UTF-8/LF hash and exact on-disk file SHA are now both recorded in the
post-hoc report. Future effective-config writes explicitly use LF.

**Attempt 3 — completed, isolated weight-decay comparison.** The only intended
modeling change from attempt 2 was AdamW weight decay `0.01` → `0.1`; the
train-only sensor standardization, seed, data/splits, architecture, learning
rate/scheduler, batch size and fixed 80-epoch budget were retained. The
attempt-2 config recorded `AdamW(weight_decay=0.01)` in one string; attempt 3
records `weight_decay: 0.1` explicitly. After normalizing that representation,
the effective configs match apart from weight decay and source provenance for
the new CLI parameter. All three arms completed 13,600 updates / 433,920
presentations each. Each arm saw the same 5,424 unique training assets per
epoch; the run total is 1,301,760 presentations across arms. The calibration
macro-NLL selector chose epochs 2, 3, and 9, respectively (10,848 / 16,272 /
48,816 presentations consumed by those selected checkpoints). The final
epoch-80 weights are separate from selected weights.

| Arm | Selected epoch | Trainable params | Selected train macro Acc / NLL | Validation Acc | Validation macro Acc | Macro NLL | Macro Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio only | 2 | 133,538 | 86.90% / 0.350 | 69.42% | 60.79% | 1.0492 | 0.5242 | 0.0988 |
| Sensor only | 3 | 96,962 | 39.13% / 1.695 | 26.25% | 31.77% | 2.0097 | 0.7942 | 0.0719 |
| Audio + sensor gated | 9 | 180,387 | 94.22% / 0.157 | 75.85% | 72.46% | 0.8908 | 0.3870 | 0.1006 |

Compared on the exact same 762 validation IDs as attempt 2, audio-only
regressed (75.07% → 69.42% sample Accuracy; 66.83% → 60.79% macro Accuracy).
Sensor-only remained weak (27.56% → 26.25% sample Accuracy). The gated model
improved (71.92% → 75.85% sample Accuracy; 64.80% → 72.46% macro Accuracy;
macro NLL 0.9455 → 0.8908; macro Brier 0.4698 → 0.3870), with ECE worsening
from 0.0643 to 0.1006. Paired bootstrap over validation trials stratified by
participant × target activity (10,000 replicates, seed 17) estimated fused
minus attempt-2 accuracy at +3.92 pp (95% interval +1.57 to +6.30) and macro
Accuracy at +7.62 pp (+4.69 to +10.57). Macro NLL delta was −0.0546 (−0.1201
to +0.0126); macro Brier delta −0.0827 (−0.1075 to −0.0577). These intervals
are conditional on the same two held-out participants, not new participants
or blind audit evidence. Checkpoint selection in each run used calibration,
not this validation set.

The paired result is an encouraging validation-only indication that stronger
weight decay helped the gated architecture, not proof that the sensor content
caused the gain. On attempt 3, shuffling sensor rows changed fused macro
Accuracy only from 72.46% to 71.16% and macro NLL from 0.8908 to 0.9261; the
audio-absent control scored 8.80% macro Accuracy, while the sensor-absent
control scored 62.81%. The shuffle result still cautions that the learned
fusion may use sensor structure weakly or as a regularizer. Audio-only also
regressed in this attempt, so the modality trade-off is material. Do not
promote this candidate as a general multimodal Decision model.

The selected fused checkpoint has a rolling train macro Accuracy of 94.22%
and validation macro Accuracy of 72.46%; this is not a fixed-example
generalization gap estimate. Calibration macro NLL reached 0.8876 at epoch 9,
was 1.3142 at epoch 20, 2.2657 at epoch 40 and 3.3506 at epoch 80 while
within-epoch train Accuracy rose from 92.85% to 100%. This is clear late
overfitting. The predeclared fixed 80-epoch training budget completed; no
performance-based early stop was used. Selected checkpoint reload on CUDA
reproduced saved predictions exactly for all three arms (max absolute
probability delta `0.0`).

| Attempt 3 selected arm | Attempt 2 → attempt 3 sample Acc | Macro Acc | Macro NLL | Macro Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Audio only | 75.07% → 69.42% | 66.83% → 60.79% | 0.9785 → 1.0492 | 0.4664 → 0.5242 | 0.1229 → 0.0988 |
| Sensor only | 27.56% → 26.25% | 30.11% → 31.77% | 2.1305 → 2.0097 | 0.8001 → 0.7942 | 0.1030 → 0.0719 |
| Audio + sensor gated | 71.92% → 75.85% | 64.80% → 72.46% | 0.9455 → 0.8908 | 0.4698 → 0.3870 | 0.0643 → 0.1006 |

## Compute and artifacts

- Local hardware: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB; no cloud compute.
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA runtime 12.4,
  NumPy 2.4.6. No model package upgrade was needed for this readout fit.
- Wall time: 1,015.6 seconds total (about 16 min 56 sec); per-arm elapsed time
  was 303.9 / 229.1 / 476.6 seconds. Peak PyTorch CUDA allocation was 54.6 /
  36.4 / 71.1 MiB. External compute cost: $0.
- Attempt-2 config canonical SHA-256:
  `ef7fde649fc889f9f20ea7d418f7fa50f24ff5527b9b0abade5fbe7d3902c797`;
  exact on-disk effective-config SHA-256:
  `40d4fcb9a0298b0868d7b493826b3ffa1b685a8b7706ad4b11a3f34e018293bd`.
  Source commit was `02cdb9d07551fa788e3c933c5c1c397ff627d378`; the training
  source-file hashes and dirty-worktree fingerprint are in the effective config.
- Attempt 3 ran locally from 2026-11-20 15:22:17 UTC through 15:42:41 UTC
  (about 20 min 25 sec elapsed; per-arm elapsed 376.9 / 302.2 / 538.4 sec).
  Peak PyTorch CUDA allocation was 54.6 / 36.4 / 71.1 MiB. RTX 3080 Laptop
  16 GiB, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6; cloud
  cost `$0`.
- Attempt-3 effective config SHA-256 (canonical and exact LF file):
  `89aeb7cede1edece23cd6a80039aeff8580b49793b8b82fd7c6e11671b45c2fe`.
  It records seed 17, fixed run parameters and the explicit `0.1` weight decay;
  source commit is `02cdb9d07551fa788e3c933c5c1c397ff627d378`, with modified
  source-file hashes captured in the artifact config.
- Training command (run from the repository root):

  ```powershell
  $env:PYTHONPATH='src'
  py -3.11 scripts/train_r2m_activity_decision.py `
    --feature-cache 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-frozen-feature-cache-v1-attempt-02' `
    --pair-index-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01' `
    --output-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-supplied-activity-decision-v1-attempt-02' `
    --epochs 80 --batch-size 32 --learning-rate 0.001 --device cuda
  ```

- Selected/final checkpoints, 240-epoch histories, per-example predictions,
  controls, effective config, exact environment, post-hoc reload and paired
  analysis all remain outside Git under the attempt-02 directory above.
  Attempt-01 failure outputs remain in the separate attempt-01 directory.
  No checkpoint or media was added to Git; no quantization or product promotion.

Attempt-3 checkpoints, 240-epoch histories, evaluation controls, exact CUDA
reload report, and the paired attempt-2 comparison remain outside Git at
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-supplied-activity-decision-v1-attempt-03`.
The cross-attempt report is `attempt03-vs-attempt02-paired-analysis-corrected.json`.

## Current interpretation and next work

Attempt 3 changes the previous conclusion: the `weight_decay=0.1` gated model
is a promising *development-validation candidate* and should be retained for a
fresh, separately admitted participant evaluation after its protocol is
specified. Do not treat its observed validation gain as generalization proof:
the evaluation has only two held-out participant clusters, calibration chose
each checkpoint, sensor shuffling barely changed metrics, audio-only
regressed, and the late learning curve overfits. Preserve all three attempts
and keep attempt 2 as the fixed comparator. Before training another model,
design an identity-grouped participant split with sufficient fresh
participants and a sensor-reliance intervention that measures paired input
dependence. Do not open sealed audit, reuse observed validation for selection,
or promote any attempt as a product Teacher. The four-modality 95–98% objective
remains unmet.

\n