# AV temporal readout scale experiment v2 — results

**Status: completed as a CPU readout experiment; no product model was trained.**

## Question and frozen design

This experiment asked whether increasing the unique Perception Test video groups
from 96 to 224 would reduce the rapid train/validation divergence in the earlier
small readout experiment, and whether keeping eight VideoPrism frame features
in sequence would help compared with mean pooling. The task was to choose which
of two same-video, same-label audio event windows matched an eight-frame video
interval. It is a narrow development task, not general audio/video understanding.

The six fixed readout arms were audio-only, pooled AV, and eight-frame LSTM AV,
each with nested 96- and 224-video training cohorts. All used seed 17, 2,048
full-batch AdamW updates, learning rate 0.002, weight decay 0.001, constant
schedule, and validation every 32 updates. The 64 validation videos were the
same for every arm. Selection was the earliest minimum validation categorical
NLL. No early stopping was used; every arm completed its full budget.

The frozen protocol is
[`R2_AV_TEMPORAL_SCALE_V2_PROTOCOL_20261021.md`](R2_AV_TEMPORAL_SCALE_V2_PROTOCOL_20261021.md).
The complete selected-step curves are in
[`R2_AV_TEMPORAL_SCALE_V2_CURVES_20261022.csv`](R2_AV_TEMPORAL_SCALE_V2_CURVES_20261022.csv).

## Source and output provenance

- Repository base commit: `1c54c32d941c76b448f8a2b500ef6bb5e290bcc4`.
- The fit script was uncommitted when run; its SHA-256 was
  `363cbfc320f391bcdfb9ba82de5653c2b9820ea338332d093a21a88671de50f4`.
- Fit output: `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\readouts-scale-v2-20261022-02`.
- Run metadata status: `PASS_FRESH_AV_TEMPORAL_SCALE_FIT`;
  `result.json` SHA-256:
  `6edf9366486a530fbb3a205d694f9201702ed5527d4b6f780d93de16764fd2c4`.
- Effective config SHA-256:
  `d1272fce8289db15042c8c960eeec743b2c11e1f62addb6746e9315b885aa17f`.
  The config file still says `RUNNING`; the terminal `run.json` and
  `result.json` both say PASS. This is a metadata-finalization defect in the
  script and is retained as generated. Do not use the effective-config status
  field as the run status. The script was fixed afterward so future runs
  finalize this field; the historical output was not edited.
- Training-history SHA-256:
  `12d0ad34a1e1e11cda48868174ca43fefa3a9fdc8d848048dc355fdb58f7d6cd`.
- Validation-predictions SHA-256:
  `6337c2dfaec981d7ea2a5bca11e88f57399e263de1e9d3729a1c6456b6d79c45`.
- Video feature result SHA-256:
  `36f719f09da86d88029a391be6a23c301e728f2d7324c2280a48e792063b8a2c`;
  pinned VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
  source revision `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Audio feature result SHA-256:
  `607746f743323a88ee895537df3009dd3198dc7b9cc21a93b6ae600c6785075e`;
  pinned EfficientAT checkpoint SHA-256:
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`;
  source revision `a425fdce92572e602a1d5634799bd9f1f2efa806`.
- Best and final checkpoint SHA-256 values are listed below. All six best
  checkpoint files were separately loaded on CPU with strict state-dict
  validation after the run. Their predictions exactly match the saved
  validation predictions (maximum score difference 0 for every arm). The
  separate audit is at
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\readouts-scale-v2-20261022-02-checkpoint-audit\audit.json`,
  SHA-256 `07c4990a3c78feacfba0b43678ebb12a971fe190962ca004e0b3be25a4b36472`.
- The original literal shell command was not persisted. The CLI invocation is
  reconstructable from the recorded inputs and parser as:

  ```powershell
  python scripts/fit_perception_av_temporal_scale.py `
    --manifest C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\task_manifest.json `
    --video-features C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\features_videoprism_per_frame_mean\result.json `
    --audio-features C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\features_efficientat\result.json `
    --output C:\CodexArtifacts\pretrained-reuse-r2\perception-av-temporal-scale-v2-20261021-01\readouts-scale-v2-20261022-02
  ```

  This is an equivalent reconstructed invocation, not a verified shell-history
  transcription.

## Split, data, and compute audit

The frozen Perception Test annotation revision was
`3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; only official TRAIN records were
used. The manifest SHA-256 is
`9147e598a4ac9e8412204b472ac6b32a019517ff682b8587379f23605672dbef`. It contains
224 unique train video groups (with the 96-group cohort nested inside) and 64
separate validation video groups. Train/validation video overlap is zero. The
media identity audit passed with 288 distinct video and 288 distinct audio
hashes, no cross-split media overlap, and no match to the 1,122 verified prior
exposure receipts. The audit SHA-256 is
`68edf4c9bf26601ee27dc7d67d450f71a5403e42a67c8b1de77458dbdd82ab30`.
Official test and sealed-audit data were not opened.

The source manifest, preflight, acquisition, validation acquisition, media
identity audit, and fresh-valid-media hashes are respectively:

| Artifact | SHA-256 |
|---|---|
| `task_manifest.json` | `9147e598a4ac9e8412204b472ac6b32a019517ff682b8587379f23605672dbef` |
| `media-preflight.json` | `7b5fec8c016fd49d0555fb455334d303b2a4a3e9f8c969d6d3d9edb68e8dbea5` |
| `media-acquisition.json` | `005d2d40410e4ee4a0e8680c100146e775ec19317d96acf2fe7d51f97ca67463` |
| `validation-media-manifest.json` | `7e24882721eaef92b6d9d3d00b4329385ddca3e25b9af2836f4f36721fda6668` |
| `validation-media-preflight.json` | `71f9309c1fb04987c66f920fd86df97bb2f9d428cf7b62f4cac5df5dfb651798` |
| `validation-media-acquisition.json` | `63ce5979264e310110e55e043d806326950972912bb4d16ab13dfeaf55093861` |
| `media-identity-audit.json` | `68edf4c9bf26601ee27dc7d67d450f71a5403e42a67c8b1de77458dbdd82ab30` |
| `fresh-valid-media-report.json` | `0927cb71f1941b30b422b98fba311373793be438d47f32b480e93ace91ae2fb0` |

Media acquisition read 4,186,443,932 bytes for train and 1,067,810,737 bytes
for validation. VideoPrism extracted 288 eight-frame feature sequences on CPU
in 675.96 seconds. EfficientAT extracted features for 288 examples / 576
candidate windows; peak GPU allocation was 43,603,456 bytes and the model state
was unchanged. The readout fit itself ran on CPU for 83.82 seconds; no GPU
training or cloud compute was used, and no external compute charge was incurred.
The fit run did not
persist Python, PyTorch, CUDA, or OS versions. At report time the host reported
Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6, Windows 11, and an Intel
i7-12700H; these are current host observations, not a captured experiment
environment lock.

## Learning curves

Train rows are full-batch repeated exposures, not independent examples. Each
96-group arm made `96 × 2,048 = 196,608` group-visits; each 224-group arm made
`224 × 2,048 = 458,752` group-visits. Across the three arms at each size, this
was 1,966,080 total group-visits. The selected checkpoint in every arm was
step 0, before any optimizer update; each completed run nevertheless reached
step 2,048 and retained separate best and final weights.

The table reports training NLL / accuracy and validation NLL / accuracy.
All validation comparisons use the same 64 videos.

| Updates | Audio 96 | Pooled AV 96 | Sequence AV 96 | Audio 224 | Pooled AV 224 | Sequence AV 224 |
|---:|---|---|---|---|---|---|
| 0 | .704 / .552; .702 / .547 | .698 / .417; .688 / .594 | .698 / .427; .689 / .609 | .701 / .540; .702 / .563 | .694 / .487; .688 / .625 | .694 / .500; .689 / .594 |
| 32 | .052 / 1.000; .926 / .484 | .011 / 1.000; 1.212 / .500 | .011 / 1.000; 1.121 / .531 | .152 / .991; .988 / .484 | .014 / 1.000; 1.250 / .547 | .017 / 1.000; 1.066 / .547 |
| 512 | .008 / 1.000; 1.542 / .516 | .007 / 1.000; 1.717 / .531 | .007 / 1.000; 1.867 / .516 | .004 / 1.000; 1.811 / .469 | .003 / 1.000; 1.990 / .562 | .003 / 1.000; 2.137 / .547 |
| 1,024 | .007 / 1.000; 1.700 / .516 | .007 / 1.000; 1.876 / .547 | .007 / 1.000; 2.059 / .516 | .003 / 1.000; 1.994 / .484 | .003 / 1.000; 2.147 / .562 | .003 / 1.000; 2.315 / .531 |
| 1,536 | .007 / 1.000; 1.797 / .516 | .007 / 1.000; 1.982 / .547 | .007 / 1.000; 2.178 / .516 | .003 / 1.000; 2.110 / .484 | .003 / 1.000; 2.249 / .547 | .003 / 1.000; 2.404 / .500 |
| 1,920 | .007 / 1.000; 1.854 / .516 | .007 / 1.000; 2.047 / .547 | .007 / 1.000; 2.251 / .516 | .003 / 1.000; 2.180 / .484 | .003 / 1.000; 2.310 / .531 | .003 / 1.000; 2.448 / .484 |
| 2,048 | .007 / 1.000; 1.871 / .516 | .007 / 1.000; 2.066 / .547 | .007 / 1.000; 2.272 / .516 | .003 / 1.000; 2.199 / .484 | .003 / 1.000; 2.328 / .531 | .003 / 1.000; 2.461 / .484 |

Every head reached 100% training accuracy by step 32 (the 224-group audio-only
head reached 99.1% at step 32 and 100% by step 128). Train NLL continued to
fall while validation NLL rose. The selector therefore kept initialization
(step 0) in all six arms. This is strong overfitting within this task and
budget. More repeated optimization did not improve the selected validation
checkpoint; increasing unique groups from 96 to 224 did not stop memorization.

## Selected and final metrics

The rows below are validation metrics at each arm's selected step (0). They
are not metrics for the trained final weights. ECE uses ten bins.

| Arm | Selected step | Accuracy | NLL | Brier | ECE-10 | Mean confidence | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-only, 96 | 0 | 0.5469 | 0.7019 | 0.5085 | 0.1212 | 0.5596 | 12,353 |
| Pooled AV, 96 | 0 | 0.5938 | 0.6885 | 0.4953 | 0.0793 | 0.5144 | 50,625 |
| Sequence AV, 96 | 0 | 0.6094 | 0.6888 | 0.4956 | 0.0951 | 0.5143 | 50,625 |
| Audio-only, 224 | 0 | 0.5625 | 0.7015 | 0.5082 | 0.1295 | 0.5609 | 12,353 |
| Pooled AV, 224 | 0 | 0.6250 | 0.6884 | 0.4952 | 0.1108 | 0.5142 | 50,625 |
| Sequence AV, 224 | 0 | 0.5938 | 0.6887 | 0.4955 | 0.0798 | 0.5140 | 50,625 |

The final step-2,048 checkpoints were retained but not selected. Their
validation `(accuracy, NLL, Brier, ECE-10)` were:

| Arm | Final metrics |
|---|---|
| Audio-only, 96 | `(0.5156, 1.8714, 0.8000, 0.3819)` |
| Pooled AV, 96 | `(0.5469, 2.0659, 0.6862, 0.3219)` |
| Sequence AV, 96 | `(0.5156, 2.2724, 0.6537, 0.3004)` |
| Audio-only, 224 | `(0.4844, 2.1993, 0.9278, 0.4557)` |
| Pooled AV, 224 | `(0.5313, 2.3284, 0.7876, 0.3591)` |
| Sequence AV, 224 | `(0.4844, 2.4613, 0.7404, 0.2958)` |

Final-step metrics are included to show the trajectory, not to select a model.
Final metrics are computed from the saved history. Each final checkpoint is
available under its arm directory alongside the selected checkpoint.

## Matched comparisons and temporal controls

Paired intervals used 5,000 resamples with seed 17 and resampled the 64 video
groups. Intervals are descriptive for this validation set; all include zero.

| Comparison (candidate − reference) | Accuracy delta (95% CI) | NLL delta (95% CI) |
|---|---:|---:|
| 224 vs 96, audio-only | +0.0156 [−0.0313, +0.0781] | −0.00040 [−0.00349, +0.00268] |
| 224 vs 96, pooled AV | +0.0313 [−0.0313, +0.0938] | −0.00008 [−0.00087, +0.00073] |
| 224 vs 96, sequence AV | −0.0156 [−0.0469, 0.0000] | −0.00014 [−0.00093, +0.00065] |
| Sequence vs pooled, 96 | +0.0156 [−0.0313, +0.0781] | +0.00034 [−0.00136, +0.00205] |
| Sequence vs pooled, 224 | −0.0313 [−0.0938, +0.0313] | +0.00027 [−0.00138, +0.00191] |

At selected step 0, masking video changed sequence accuracy by −3.13 points
(96-group arm) and −1.56 points (224-group arm); the corresponding intervals
included zero. Reversing time changed accuracy by −1.56 points in the 96-group
arm (interval ending at zero) and 0 points in the 224-group arm. Masking audio
gave chance-level 46.88% accuracy in both arms. Because the selected weights
are initialization and all these intervals include zero, these controls do
not demonstrate learned temporal-order sensitivity or audiovisual
complementarity.

## Retained attempts and integrity notes

Attempt `readouts-scale-v2-20261022-01` failed before completing an optimizer
step: the audio-only arm passed a video tensor to a scorer whose forward method
accepts audio and target only. Its `run.json` reports `FAILED_SCALE_FIT`,
`TypeError`, and the error text. Its normalization files were retained. The
failure was fixed in `_data_for_arm` and covered by
`test_audio_only_arm_hides_video_tensor_and_keeps_targets`; attempt 01 was not
reused or overwritten. Attempt 02 is the successful, separate output above.

The successful `effective_config.json` retains a stale `RUNNING` status even
though `run.json` and `result.json` are terminal PASS. The script's lifecycle
status finalization was fixed after the experiment and is covered by a unit
test; the original file remains untouched. Apart from that lifecycle field,
the config records the seed, optimizer, LR, weight decay, update count,
normalization, selector, cohort sizes, feature source revisions, and checkpoint
pins. The exact original shell command and Python/PyTorch package lock were not
persisted; these omissions are explicit, not inferred away.

## Interpretation and next step

**Scale hypothesis:** NOT SUPPORTED WITHIN THIS BUDGET for this readout/task.
All arms memorized their training groups rapidly, validation NLL degraded, and
224 groups did not yield a clear selected-checkpoint gain over 96. The
sequence-preserving LSTM readout did not outperform matched mean pooling; its
paired intervals include zero and its selected checkpoint was untrained.

This does not establish that more data, recurrence, or pretrained multimodal
features cannot work. The measured task is 64 validation videos from one
source, the head is small and trained with full-batch repetition, and
validation both selected checkpoints and served as the sole evaluation. The
results point to task/head/optimization and label construction as unresolved
factors; they do not isolate which one caused overfitting. There is no basis to
promote any candidate or claim the broader four-modality quality target is
closer.

Before another fit, inspect the task's negative construction and train/valid
feature-label dependence, then use a training schedule that avoids 32-step
memorization or a more structured split/task. Freeze a new protocol and fresh
video-group validation cohort before model selection. Do not re-tune on these
64 validation videos. Teacher v1, prior v2 artifacts, external media and all
checkpoints were left unchanged; no main merge or product promotion occurred.

The six best/final checkpoint hashes (best is always step 0) are:

| Arm | Best checkpoint SHA-256 | Final checkpoint SHA-256 |
|---|---|---|
| Audio-only, 96 | `1603c525c47452d1f8a712f5f8f2ba8170e9e42bc21d233c2fd2e02d0e29710e` | `4e56097b7d9a91baf48d619e23c53b631f17bf8cb5d492d340e93db40f7d8f40` |
| Pooled AV, 96 | `0151f8d7d5752872af38e016de2720f0502ae1ec0022126109bcbb44ebb2ec7d` | `871940d6ce8efe2d98acff460f98d0804c331e01d2ae2947291a8492f49c48f0` |
| Sequence AV, 96 | `d2885150acc965af6405bf2f280e1c82c31118a95d7e89d8558917aa06d49ec3` | `2651e443ac200a340d85280ce1ed7aaec11f0418d2dc19a214e7d2f64d0aaae2` |
| Audio-only, 224 | `f839f14fbb99c0a226b7c74000577429cbf155b2039ee73510e621f8cb81dbbb` | `3aacaa9118ffd4cd0edadf252b5b8543b799a77ee0e9e51b2107f99bf29faee8` |
| Pooled AV, 224 | `62ac356f6e7b27f76f8dc10737f068ee58e9eb3ac196d0a8d6d279491286320e` | `46ecef758cc14a4855efa3b76630e057d51209f2f7360d5058e5423c6d459068` |
| Sequence AV, 224 | `e9b6f99e1a2a8f703e4dc9c9488c0c171f2c104c69d25ea6d28390ec27efea75` | `5893300b2da1aa0a7f73388791fa903fe82732042db56b07f4a0a4ddfb02347f` |
