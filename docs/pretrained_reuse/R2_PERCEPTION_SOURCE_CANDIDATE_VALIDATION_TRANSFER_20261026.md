# Perception Test source-candidate fixed-checkpoint validation transfer — 2026-10-26

## Result and scope

This report completes the previously pending, fixed-checkpoint evaluation on
the 138 eligible records in the already inspected official-validation cohort.
It does not replace the earlier integrity report, and it does not describe a
blind test. The cohort was selected and inspected in a previous visibility
diagnostic. No fitting, calibration, model selection, or sealed-audit access
was performed here.

**Joint audio/video result:** 52.17% set Accuracy, 1.4637 set-NLL, 0.5461
multiclass Brier, and 0.1243 top-1 ECE. The 95% video-group bootstrap interval
for Accuracy is 43.48–60.14% (10,000 resamples, seed 17).

The evidence supports that this particular scorer uses visual features and
that mismatched audio harms predictions. It does **not** establish an additive
audio/video accuracy gain over video alone: the joint Accuracy is lower than
the video-only point estimate, while its NLL/Brier point estimates are slightly
better and the paired intervals for those differences include zero. Audio-only
performance is weak. This is narrow oracle-assisted source-retrieval evidence,
not a general multimodal Decision model result and not evidence for text or
image quality.

## Frozen inputs and model

- Dataset code/revision: Perception Test source revision
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
- Frozen source-candidate record file SHA-256:
  `4a15269531ef9a2a1bccdab731fa23e30cd3754850fc72fd35028b9986bd0844`.
- Corpus manifest SHA-256:
  `f85afede73fe1d90bfed0dca7fe69dd7381b8a1102d06be531d7d4d27bcc3d5f`.
- Selection SHA-256:
  `10cdc277cf417acba7e50245d8e0d189a6f60ce467c8d87dade42a41cb05e6d9`.
- Candidate cohort: 138 records from 138 distinct validation videos, 79
  `none_visible` and 59 with at least one visible source. The 200-row corpus
  also contains 61 held rows and one duplicate; only the 138 preselected
  candidate rows entered extraction and scoring.
- Existing model-fitting source: 2,593 training events from 320 videos and
  606 internal-development events from 80 videos. The fixed readout was
  selected at epoch 7 of an 80-epoch run using development set-NLL. The
  training-label SHA-256 is
  `c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d`; the
  internal-development-label SHA-256 is
  `701bd1ab203e54fae5fcb7cc8e5987c2d903bd23b5d2fbf736c2132e953ebcda`.
- Frozen readout checkpoint SHA-256:
  `63390ba7532348bbd600f64ed796e92a3daf13b1f44ba0b18e83b1fd304780bf`.
  It is the existing epoch-7 state, not a new or selected-on-validation
  checkpoint. The prior internal-development result (56.93% set Accuracy /
  1.2748 NLL) is a different, exposed development cohort and is not combined
  with the current validation metrics.
- Encoder checkpoints: VideoPrism-B SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
  (114,365,184 parameters); EfficientAT SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`
  (1,961,239 parameters). The readout has 311,809 trainable parameters. The
  component sum is approximately 116,638,232 unquantized parameters. No
  quantization or encoder training was performed.
- Train/validation overlap check: zero video groups, as recorded in the
  previously pinned corpus integrity audit. The current validation rows are
  unique by video. One row (`video_2483`) records an upstream cut at frame 400;
  its largest requested frame is 300 and all requested frames were decoded.

The validation candidate count ranges from 2 to 43 (mean 14.64); the training
maximum was 35. The set-level uniform-random expected top-1 hit rate, weighted
by each event's number of valid targets and choices, is 9.71%. One validation
event therefore has more candidates than appeared in training. The wide and
shifted candidate-count distribution is a task limitation to keep visible in
any interpretation.

## Feature extraction and fixed evaluation

The label-free plan was built only after matching the fixed corpus, selection,
acquisition receipts, and media preflight. It covers 138 records/138 videos,
431 unique annotated frames, and 152 VideoPrism temporal windows. The plan
contains candidate geometry and event audio intervals, but no targets.

- Derived acquisition manifest SHA-256:
  `329aa000860b6005e76b97d6aa06a169beb016b9db8e1893410513ff62ac52fe`.
- Media preflight SHA-256:
  `9e17e0dd19cf7b23d423dd59ee42dbe54e878bcc339ee9c8b837e4516a8842ca`.
- Label-free feature-plan SHA-256:
  `02181e06bab59a98f62c27e0a39c83f5a92d55ef750bed455331a5d13ed64c0e`.
- Video features: 4,145 candidate/frame ROI vectors of dimension 768;
  SHA-256 `758dc8a3358b2a3b1a0b4f41f5990e42b5caebc111833ac1c62f54192b583d59`.
  Corrected result metadata SHA-256:
  `85b4e64f652e21c74c5be24e7491af05bf820e0fc44a3e69c4ec78328e78ad38`.
  The CPU extraction took 412.46 seconds with the pinned VideoPrism-B
  checkpoint/source.
- Audio features: 267 unique 2-second windows/references of dimension 384;
  SHA-256 `2c56db22870df65c76974c845a62b39c30125820cfb099e0d23db5b1038e0e51`.
  Result metadata SHA-256:
  `beaa3b3545a608cce7de38549aca57814788b0039681706a174b4042f0318c58`.
  The existing WAV media was decoded with the Python standard-library PCM
  reader and resampled to 32 kHz using the already installed torchaudio.
  EfficientAT extraction took 11.57 seconds on the RTX 3080 Laptop GPU; peak
  allocated/reserved memory was 77,399,040 / 90,177,536 bytes. The model-state
  hashes before/after feature extraction matched.
- The first audio attempt used a Python environment without `torchaudio` and
  failed before extraction. HotUGPU released its reservation; the retry used
  the existing `perception-test-av-v1` environment (Python 3.11.9,
  PyTorch/torchaudio 2.6.0+cu124), with no package changes. The guarded runner
  waited for the GPU owner to yield and then launched; no process was manually
  stopped. VideoPrism ran in its pinned Python 3.11.16/JAX 0.6.2/TensorFlow
  2.18.1 CPU environment. Preflight used PyAV 18.1.0. Evaluation used CPU
  inference from the fixed readout state. No cloud compute or compute fees.

The fixed-checkpoint scorer was evaluated with the same labels and features in
five input conditions. `video_only` zeros audio; `audio_only` zeros video;
`mismatched_audio` cyclically assigns each video's audio to the next validation
video in sorted record order; `reverse_video_temporal_delta` negates the
chronological last-minus-first visual component. These are diagnostics, not
candidate selection runs.

| Condition | Set Accuracy | Set-NLL | Brier | Top-1 ECE | Visible-source hit@1 | None-visible Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint audio + video | 52.17% | 1.4637 | 0.5461 | 0.1243 | 30.51% | 68.35% |
| Video only (zero audio) | 55.07% | 1.4804 | 0.5481 | 0.1551 | 6.78% | 91.14% |
| Audio only (zero video) | 13.77% | 2.4644 | 0.7796 | 0.0680 | 8.47% | 17.72% |
| Mismatched audio | 41.30% | 1.9739 | 0.7223 | 0.2366 | 22.03% | 55.70% |
| Reversed visual temporal delta | 55.07% | 1.4493 | 0.5490 | 0.0867 | 30.51% | 73.42% |
| Training-frequency option/none prior only | 57.25% | 1.7369 | 0.5845 | 0.2572 | 0.00% | 100.00% |

The option-only baseline uses training-label target frequencies by candidate
slot plus the `none_visible` frequency, with add-one smoothing. It loads no
media, features, or readout checkpoint and uses no validation labels to fit the
prior. It predicts `none_visible` on all 138 validation events. Its higher
overall Accuracy than the joint model is therefore a class-prior artifact, not
successful source localization: its visible-source hit@1 is zero. This is why
overall Accuracy alone is not an adequate success measure for this task.

## Paired uncertainty and interpretation

All intervals below use 10,000 paired bootstrap resamples over the same 138
unique video groups with seed 17. Deltas are comparison minus joint; positive
NLL/Brier deltas are worse.

| Comparison | Accuracy delta (95% interval) | NLL delta (95% interval) | Brier delta (95% interval) |
| --- | ---: | ---: | ---: |
| Video only | +2.90 pp (−5.07, +11.59) | +0.0166 (−0.1211, +0.1480) | +0.0020 (−0.0634, +0.0649) |
| Audio only | −38.41 pp (−49.28, −27.54) | +1.0006 (+0.7132, +1.2680) | +0.2336 (+0.1411, +0.3227) |
| Mismatched audio | −10.87 pp (−19.57, −2.17) | +0.5101 (+0.3097, +0.7219) | +0.1762 (+0.0901, +0.2616) |
| Reversed visual temporal delta | +2.90 pp (−4.35, +9.42) | −0.0144 (−0.1732, +0.1449) | +0.0029 (−0.0555, +0.0622) |
| Option-only prior | +5.07 pp (−4.35, +13.77) | +0.2732 (+0.0392, +0.4951) | +0.0385 (−0.0399, +0.1138) |

The video-zero ablation and its bootstrap show that video is necessary for this
model's useful source localization. Swapping audio across videos significantly
reduces Accuracy and worsens NLL/Brier, so audio alignment carries information
the scorer uses. However, joint-vs-video-only intervals include zero; current
evidence does not establish a reliable aggregate gain from adding audio. The
reversed temporal delta also does not reduce performance, so this experiment
does not establish that chronological direction is useful to this readout.
The training-frequency prior's Accuracy illustrates that the selected cohort's
label composition can dominate top-1 Accuracy while failing the visible-source
part of the task.

## Artifacts and reproducibility

The immutable run artifacts are outside Git under:

`C:\CodexArtifacts\pretrained-reuse-r2\perception-source-candidate-validation-transfer-20261026-attempt-07\`

This directory contains the derived acquisition manifest, media preflight,
feature plan, CPU VideoPrism features, guarded EfficientAT features, fixed
checkpoint predictions/metrics, five diagnostic prediction files, the paired
bootstrap JSON, and the option-only baseline predictions/metrics. Important
hashes:

- Diagnostic-controls result:
  `79b2626fe76492c4f31b3e7f2805481d4770dc86be6a469fff0d357e8a6410aa`.
- Paired-differences JSON SHA-256:
  `15e1c65c30fd526be82afb8f3a13bcf0332af19583f7a70e5d00d49b5c3d408c`.
- Option-prior result SHA-256:
  `9a62344f326991936d5af694d00a5b4eb1c4eda2dad9113bc86e4bcec43c2b97`;
  its prediction SHA-256 is recorded inside the file.
- Option-prior versus joint paired-differences JSON SHA-256:
  `98c77a7316c0f7a133282002bdc3226f5623b13984ebed81585f05ef74c86f6a`.
- Video and audio result/index hashes are recorded in their respective
  `result.json` and `run.json` files.

The VideoPrism extractor's first result contained a stale hard-coded
`validation_scope` string, despite recording `split=validation` and completing
all 138 videos. The raw `run.json` and `result.json` are preserved as
`run-before-scope-correction.json` and `result-before-scope-correction.json`;
the corrected metadata and raw/corrected hashes are recorded in
`video-features/scope-correction.json`. No feature arrays or indices changed.

New code and tests were added on the existing documentation branch. Existing
Teacher checkpoints, source corpora, sealed audit, unrelated worktrees, PRs,
and live processes were not changed. No model was trained and no checkpoint
was selected from this validation cohort.

## R2 status and next action

This evidence closes the previously pending extraction/evaluation step for
this source-candidate task, but it does **not** close the Roadmap's broader
R2 multimodal Decision gate. The task is oracle-assisted, the validation cohort
is non-blind and selected, candidate counts shift beyond training, the
option-only predictor reaches 57.25% by always choosing none, and the joint
model does not beat video-only Accuracy. Keep this readout diagnostic-only.
Do not add a fusion module or fit another head on these exposed validation IDs.
The next quality experiment needs a fresh grouped cohort and a task whose
useful answer demonstrably depends on both streams; it must retain an
option/metadata-only baseline and the same unimodal/mismatched-input controls.
