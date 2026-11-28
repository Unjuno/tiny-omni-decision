# Physion++ VideoPrism frozen-feature Path A — 2026-11-28

## Result scope

This is one seed-17, task-specific binary Decision experiment on the existing
Physion++ `fullclip-v1` train/development split. The fixed question is “Did the
red object contact the yellow-marked target at any point in the clip?” and the
options are `[No, Yes]`. It tests whether one frozen video representation can
support this single simulated contact task. It does not test general video
understanding, language variation, other modalities, or the product target.

The development validation was used for 200-epoch checkpoint selection by
minimum NLL. It is exposed development data, not a blind or independent audit.
The publisher test/sealed data was not loaded. The blinded human review of
label observability remains incomplete.

## Data and representation

The train and development manifests contain 652 and 148 records, respectively.
They have 492 and 123 scenario-plus-scene-seed groups. The split audit reports
zero overlap for record IDs, source IDs, asset groups, media paths, media
SHA-256s and normalized content. The 652 train labels were independently
matched to official stimulus metadata and contact timelines. These checks
establish serialization and split integrity; they do not establish that a
person can infer every label from the RGB frames.

Video features use frozen `google/videoprism-base-f16r288` revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, 8
uniform endpoint-inclusive frames at 288 pixels, and mean pooling to 768
dimensions. Both train and development features were extracted on CPU in the
pinned JAX environment. The readout receives each video vector and one-hot
candidate identity; it has 98,817 trainable parameters. Estimated resident
dense parameters are 114,464,001, including the frozen 114,365,184-parameter
encoder.

## Training and evaluation

The seed-17 readout ran 200 epochs without early stopping, with AdamW, learning
rate `0.001`, weight decay `0.001`, batch size 32, and per-question candidate
order randomization. The best checkpoint was epoch 55; the final epoch-200
checkpoint is preserved separately. Training ran on CPU for 8.66 seconds.
Train feature extraction took 827.65 seconds; full 148-row development feature
extraction took 197.31 seconds. The 2-row validation smoke took 16.78 seconds.
No cloud compute or GPU was used by this experiment.

| Evaluation | Accuracy | NLL | Brier | ECE-15 | Count |
|---|---:|---:|---:|---:|---:|
| Selected checkpoint, development validation | 0.7027 | 0.6019 | 0.4114 | 0.0723 | 148 |
| Selected checkpoint, train fit | 0.7285 | 0.5361 | 0.3610 | 0.0481 | 652 |
| Training-label prior control | 0.5541 | 0.6896 | 0.4965 | 0.0341 | 148 |
| Selected checkpoint, zero video features | 0.5541 | 0.6893 | 0.4961 | 0.0312 | 148 |
| Selected checkpoint, permuted validation videos (seed 17017) | 0.5270 | 0.8514 | 0.6025 | 0.2179 | 148 |

Candidate-order reversal changed logits by at most `5.96e-8`, consistent with
permutation equivariance. The train-to-validation Accuracy gap is 2.58
percentage points. The feature ablations indicate that this selected readout
uses video information on this development cohort: removing video returns
performance to the label-prior level, and permuting videos worsens NLL and
Brier.

A paired bootstrap resampled the 123 validation source-asset groups 10,000
times with replacement (PCG64 seed `20261128`) and retained all rows in each
sampled group. Selected-minus-zero-video differences were +14.86 percentage
points Accuracy (95% descriptive interval +5.59 to +24.34 pp), −0.0874 NLL
(−0.1575 to −0.0107), and −0.0847 Brier (−0.1421 to −0.0235). Because this
same exposed validation set selected the checkpoint, these intervals do not
correct for selection bias and are not independent evidence.

## Data-first interpretation and next action

This is a positive, limited video-signal result, not evidence that the data
pipeline or model has reached the intended quality. Important unresolved data
questions remain: a blinded human observability review has not been completed;
only 63.6% of positive train clips include an explicitly positive contact frame
among the sampled 8 frames (76.7% within ±4 frames); and some recorded
scenario/factor strata are sparse or single-class. Official metadata agreement
cannot substitute for visual observability. The existing low-level matched-pair
screen found no corrected differences across ten coarse render statistics,
but that does not rule out higher-order shortcuts.

Do not respond to this video-task uncertainty with a broad BERT sweep. The
question and candidate wording are fixed, so a text encoder cannot address
frame/event coverage or visual label observability. Earlier BERT evidence is
task-specific text evidence, not a remedy for this video data limitation. The
next useful gate is the pending blinded observability review and a stronger,
predeclared temporal-coverage/data-quality study. Keep the existing 8-frame
result frozen; any changed sampling policy is a distinct experiment. Do not
open sealed/test data or promote this checkpoint based on this validation.

## Reproduction and artifacts

Source commit: `2c8bb8fad88358df6a77ea950e0f964dd7da9f49`.

- Config: `configs/pretrained_reuse/path_a_video_videoprism_physionpp_contact_seed17.yaml`
- Trainer: `scripts/train_physionpp_videoprism_path_a.py`
- Validation extractor: `scripts/extract_physionpp_validation_videoprism_features.py`
- Focused validation extractor tests: 4 passed; Ruff passed.
- Train feature cache SHA-256: `32846b085df7642d9190df8f553a58496d275f8580efe6c6a8adfa3a83217360`
- Development feature cache SHA-256: `065592b9d1e01647f88b93f7da811afd07e1d40a55cd9518877ce1e1724865a4`
- Development feature report SHA-256: `c8788195a528ba18211ffb2c3da2881fc2aebb87029dfefdbf945ac7bdd5eaa6`
- Run report SHA-256: `45d92e4203c42c8e6b3f30b9b9ad0abdd67c054ed45718c7c19573c1cb1ca784`
- Best checkpoint SHA-256: `d242f3994e3cfc85683b31971bf861ddd310bda7c94683e650d21c1852081ffb`
- Final checkpoint SHA-256: `72f3f0f599222acd674a683d27b17bfe8a5f0d321707e65b19f8c14e180b1b53`
- Learning history SHA-256: `7cfb52d2b4f6498022cd87b72dd07a7569e88ed966f1549383919e44dd5f43aa`
- Validation predictions SHA-256: `7295a8029984a0f9390c82aaa202e2f70f4d14f9cb8e02b6d529b60704730f4c`
- Paired bootstrap SHA-256: `8974c0d9e5d01f8129390db5580eb0d4d4aa92a5644ab80f96edc777919b6d33`
- Local run output: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-video-path-a-seed17-attempt-01`
- Local validation cache: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-validation-videoprism-features-20261128-attempt-03`

No checkpoint, corpus, or prior run was overwritten. The full reports and
predictions remain in the local artifact store; no model weights or media were
added to Git.
