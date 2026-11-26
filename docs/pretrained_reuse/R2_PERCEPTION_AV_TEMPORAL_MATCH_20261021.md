# Perception Test same-video temporal match: exploratory result

## Result

This is a development-only audio/video matching diagnostic. It does **not** close
R2 M-2, demonstrate general audiovisual reasoning, or support adopting a fusion
gate. The task used an algorithmically defined timestamp target rather than
independently adjudicated evidence that a human decision requires both inputs.
The validation source and related labels had already been inspected, so the
results are not blind and were not used as a sealed audit.

The pooled joint readout scored **47.60%** validation choice Accuracy, below the
audio-only readout at **50.80%**. A paired bootstrap over 313 video groups gave
joint-minus-audio Accuracy **−3.19 percentage points**, with a 95% percentile
interval of **[−7.99, +1.28] points** (2,000 resamples, seed 17). Video-only
produced identical scores for both choices; its tie-adjusted choice Accuracy is
50% (a plain argmax that always chooses option 0 would be 48.24% because option
0 was correct in 151/313 examples).

A separate 19,009-parameter conditional-gate readout completed all 2,048 CPU
updates. Training Accuracy reached 100% by update 32. Validation NLL was best at
step 0, so the fixed selector chose the untrained initialization. The final
step's Accuracy was 51.76%, but NLL/Brier/ECE had degraded to 1.1862/0.3416/0.2976.
This is direct overfitting evidence for this small readout and dataset; it does
not establish a limit on gated or temporal architectures generally.

## Frozen task construction

Each example contains an 8-frame, 2-second video interval centered on an anchor
sound event, plus two 2-second audio candidates from the same source video and
the same sound-label class. One candidate is the anchor event's synchronized
audio; the hard negative is a different event of the same label, with event
centers at least 2.25 seconds apart. The target is the synchronized candidate.
Candidate order is deterministic and balanced by a stable hash. Every video
contributes one example and stays wholly in one split.

The task builder used only Perception Test official train and validation
annotations at source revision `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
It combined pairable rows from the prior fixed cohort with 100 additional
official-train videos and a previously selected validation extension:

| Split | Video groups | Candidate audio windows |
|---|---:|---:|
| Train | 190 | 380 |
| Validation | 313 | 626 |

The validation is development-only. Its class mix is uneven: sound label 7 has
134 rows, while labels 4, 5, 6, 8 and 10 have 1–10 each. The training split has
190 groups and label counts `0:17, 1:21, 4:13, 5:12, 6:15, 7:55, 8:14, 10:14,
11:29`. No official test annotations or sealed audit data were opened.

Integrity checks found zero train/validation overlap by video ID, raw video SHA,
raw audio SHA, or normalized 32 kHz audio-window PCM SHA. The 100 additional
train videos required 200 media files (2,070,637,225 expanded bytes). The
download used the pinned archive size, ETag, and HTTP range checks; each member
was checked against its declared ZIP size and CRC plus a saved SHA-256. One
initial size preflight double-counted already acquired media; it is marked
`SUPERSEDED` in the external attempt directory, and the corrected preflight and
acquisition record are preserved there.

The task manifest SHA-256 is `74a9fe9cc060ad383916fa327b8d4fe99d677082467fb64ff1c65b3824f4cb97`.
Its source annotation/cohort hashes and all experiment result hashes are in the
[machine-readable receipt](R2_PERCEPTION_AV_TEMPORAL_MATCH_20261021.json).

## Models and controls

Two existing encoders were frozen and used only to make features:

- VideoPrism-B: 114,365,184 parameters, pinned checkpoint SHA
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`. It
  processes eight sampled frames and the readout uses the mean pooled token
  feature. Exact-window prior features were reused for 221 examples; 282 were
  extracted on CPU in 697.5 seconds.
- EfficientAT DyMN: 1,961,239 parameters, pinned checkpoint SHA
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`. It
  creates a 384-dimensional feature for each candidate's 2-second crop. All
  1,006 windows were processed under the cooperative HotUGPU reservation in
  32.9 seconds; peak allocated/reserved GPU memory was 43,603,456 / 67,108,864
  bytes. No GPU training, cloud compute, or purchase was used.

The first readout comparison fit three fixed CPU logistic heads (seed 17) on
train-only PCA features: audio-only, video-only and joint
`[audio PCA-32, video PCA-32, elementwise product]`. The joint model had 97
logistic parameters; its PCA transforms were also saved. The conditional-gate
candidate projected each pooled modality to 16 dimensions, applied a
sample-and-option-conditioned sigmoid gate, and scored each choice. It used
AdamW, learning rate 0.002, weight decay 0.001 and a full batch per update. The
run completed without early stopping; its checkpoint selector was minimum
validation categorical NLL over step 0 and every 32 updates. The selected step
and final step checkpoints and full history are saved outside Git.

### Validation metrics

| Readout/control | Accuracy | NLL | Brier | ECE-10 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Audio-only | 0.5080 | 0.6931 | 0.2500 | 0.0051 | 0.5028 |
| Video-only | 0.5000* | 0.6931 | 0.2500 | 0.0000* | 0.5000 |
| Pooled joint | 0.4760 | 0.6934 | 0.2501 | 0.0275 | 0.5035 |
| Pooled joint, video shuffled within sound label | 0.5112 | 0.6927 | 0.2498 | 0.0079 | 0.5032 |
| Pooled joint, audio pairs shuffled within sound label | 0.4888 | 0.6926 | 0.2497 | 0.0145 | 0.5033 |
| Pooled joint, video mean-imputed | 0.5144 | 0.6932 | 0.2500 | 0.0115 | 0.5028 |
| Conditional gate, selected step 0 | 0.4696 | 0.6985 | 0.2527 | 0.0624 | 0.5321 |
| Conditional gate, final step 2048 | 0.5176 | 1.1862 | 0.3416 | 0.2976 | — |

`*` Video-only emits equal scores for both options. The table reports expected
choice Accuracy under a fair tie, and the corresponding tie-aware ECE. The
saved deterministic argmax metric is 0.4824/0.0176 because it always selects
option 0; the `choice_accuracy` field in the output records the tie-adjusted
value.

Shuffling the paired audio or video features within sound-label groups did not
hurt the pooled joint readout. Replacing video with its train mean also did not
hurt it. These controls provide no evidence that the pooled joint candidate
uses correct cross-modal correspondence. They are development controls on a
previously inspected validation set, not a causal or blind audit.

### Conditional-gate learning curve

| Updates | Train Accuracy | Validation Accuracy | Validation NLL | Brier | ECE-10 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4947 | 0.4696 | 0.6985 | 0.2527 | 0.0624 |
| 32 | 1.0000 | 0.5112 | 0.7425 | 0.2714 | 0.1182 |
| 64 | 1.0000 | 0.5431 | 0.7965 | 0.2868 | 0.1562 |
| 128 | 1.0000 | 0.5367 | 0.8693 | 0.3027 | 0.1972 |
| 256 | 1.0000 | 0.5240 | 0.9367 | 0.3127 | 0.2289 |
| 512 | 1.0000 | 0.5304 | 1.0121 | 0.3223 | 0.2448 |
| 1,024 | 1.0000 | 0.5335 | 1.0936 | 0.3304 | 0.2606 |
| 1,536 | 1.0000 | 0.5208 | 1.1447 | 0.3366 | 0.2862 |
| 1,792 | 1.0000 | 0.5176 | 1.1669 | 0.3394 | 0.2939 |
| 2,048 | 1.0000 | 0.5176 | 1.1862 | 0.3416 | 0.2976 |

The gate selected step 0 by its predeclared NLL rule; the validation point at
step 64 has the highest Accuracy but much worse NLL and was not selected. The
selected gate model is therefore the initialization, not a trained improvement.
A deterministic replay reproduced all predictions and curve metrics exactly;
best and final checkpoint payloads were saved and their tensors load as finite.

## Interpretation and next gate

**Finding:** neither the pooled joint readout nor this conditional gate supports
useful audio/video complementarity on this task. The current setup is particularly
limited because both encoders are reduced to one pooled vector; temporal token
sequences were not passed into the gate. There is also only one trial per video,
190 train groups, and uneven sound-label support. These facts prevent separating
data scarcity, task ambiguity and lost temporal representation. Do not call the
result a model-family ceiling or claim that a time-aware architecture was tested.

Do not tune more heads or seeds against this validation. A future M-2 attempt
needs a source whose labels are independently verified as requiring both
modalities, adequate video-grouped training coverage, sequence-level features,
matched unimodal baselines, and predeclared missing/shuffled/stale/timing-offset
controls. Keep R2 M-2 open until that evidence exists. The product Teacher,
sealed audit, existing corpora and previous checkpoints were not modified.

## Reproduction and artifacts

Implementation scripts and CPU tests are in this repository. The frozen media,
features, saved predictions, readout checkpoints and training history are kept
outside Git under:

`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-match-preflight-01`

The source/run hashes, per-label metrics, bootstrap procedure, checkpoint IDs,
compute summary and full result references are recorded in the adjacent JSON
receipt. No media or large model checkpoint is committed.
