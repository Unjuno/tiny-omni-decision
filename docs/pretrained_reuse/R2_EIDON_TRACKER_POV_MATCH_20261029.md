# Eidon Tracker POV paired video/IMU match diagnostic — 2026-10-29

## Scope and result

This is one narrow R2-M diagnostic, not a general world-understanding or product
Decision evaluation. For each query, a 5-second first-person video window is
paired with four 5-second torso-relative IMU pose sequences from the same
contributor and task. The target is the sensor sequence from the same recording
as the query video. Candidate order is randomized deterministically with seed
17. There is no task name, contributor ID, recording ID, or text question in the
model input. A correct prediction therefore needs video-to-sensor correspondence
for this retrieval task. It does not establish that both modalities are needed
for ordinary activity decisions.

The result did **not** pass the R2-M multi-input gate. On 24 development queries
from only 3 held-out contributors and 6 contributor/task groups, both the
pooled and LSTM readouts selected 7/24 correct (29.2%; four-way chance is 25%).
The LSTM's minimum development NLL was 1.3842 versus 1.3863 for uniform
probabilities; pooled NLL was 1.3870. The LSTM's NLL difference is small and was
selected on this same six-group development set. A 10,000-draw paired cluster
bootstrap gives an accuracy delta versus expected chance of +4.17 percentage
points, 95% interval [−8.33, +16.67] pp. It is too uncertain to claim a
reliable gain.

Input interventions did not consistently show that both streams were required.
The LSTM's sensor-time-reversed control scored 8/24 (33.3%) and NLL 1.3837,
slightly better than the matched sequence. Reordering options preserved the
prediction exactly, as required. Rotating candidate sensor features reduced
LSTM accuracy to 25.0% and raised NLL to 1.3885, a small change on this sample.
Removing video or sensor produced near-uniform probabilities. These results do
not demonstrate useful temporal or cross-modal dependence.

The fixed 2,000-epoch run found no delayed generalization improvement. Both
readouts reached 100% training accuracy by epoch 100 while development quality
deteriorated. At epoch 2,000, pooled development Accuracy/NLL were 20.8%/3.511;
LSTM values were 12.5%/3.263. The minimum-development-NLL checkpoints remained
at epoch 1 (pooled) and epoch 5 (LSTM). This is evidence of overfitting for these
readouts and this tiny split, not a claim that recurrent models or the source
cannot improve with different data or training.

## Source and rights

- Video repository: [`eidon-ai/tracker-pov` at revision `47c55ccc6d0308894ab482009c6fa91569057a08`](https://huggingface.co/datasets/eidon-ai/tracker-pov/tree/47c55ccc6d0308894ab482009c6fa91569057a08)
- IMU repository: [`eidon-ai/tracker-pov-imu` at revision `97d4441a7e915d2ad4b5988a43ebddd79a542aea`](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu/tree/97d4441a7e915d2ad4b5988a43ebddd79a542aea)
- Video metadata Parquet SHA-256: `d450018059bf6696ad793bab40fcb19fc49dcded8700df09047b2d508a8741e1`
- IMU `shard_index.json` SHA-256: `1b4d6f88313d779aba2c1f451446906cfb2992467abcac83b0a6f640405f7855`
- The official cards identify CC BY 4.0. The video is unaltered egocentric footage recorded in private homes; incidental faces, screens, documents, and interiors are possible and are not blurred. No raw clips or frames were added to Git. Only fixed-size feature caches and private run records were written outside the repository. See the [video card's provenance and consent](https://huggingface.co/datasets/eidon-ai/tracker-pov#provenance-and-consent) and the [IMU card](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu).

The source publishes one `train` split, so a new contributor-held-out split was
constructed from the frozen local metadata and asset hashes. The feature job
read only the frozen train/development manifest; it did not open the separate
sealed-audit manifest or evaluate any audit records. No paid compute was used.

## Frozen data and quality gate

The manifest is outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/eidon-tracker-pov-screen-20261028-attempt-01/train-development-manifest-private.json`,
SHA-256 `bec58d3e35d12542aa8734b294cf4f8f066caf18568f23e2d5c2be1ac3c48fdb`.
The frozen candidate set held 136 train and 32 development videos. All were
metadata `valid`, had seven sensor slots, and had a chest reference. `has_raw_motion`
was not used as a filter: it marks opt-in accel/gyro/magnetometer availability,
while orientation quaternions are published throughout the dataset. Only
quaternions were used here.

During extraction, six recordings failed the existing unit-quaternion tolerance
(absolute norm error at most `1e-3`). Four train and two development
contributor/task groups were quarantined in full so every remaining query still
has exactly four choices. No values were normalized, imputed, or repaired.
One affected window had quaternion norms as large as 9.26. This uniform
record-level check yielded:

| Split | Candidate videos | Usable videos | Complete groups | Held-out contributors | Task counts (cleaning / cooking / laundry) |
|---|---:|---:|---:|---:|---|
| Train | 136 | 120 | 30 | 14 | 40 / 32 / 48 |
| Development | 32 | 24 | 6 | 3 | 8 / 4 / 12 |

Train and development share no contributor or video asset. The development set
is very small and task-imbalanced; cooking has only four queries. It is not an
independent final test and is now used for development/checkpoint selection.

## Frozen representations and extraction

- Video: Google VideoPrism-B `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`, 114,365,184 frozen parameters; checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`; official source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Each selected video window is centered in the video and spans 5 seconds. Eight uniformly spaced RGB frames are decoded using bounded HTTP byte ranges. The upstream resize-short-side, 288-pixel center crop, and float RGB preprocessing are preserved. VideoPrism returns 2,048 tokens; 256 spatial tokens are averaged separately for each of 8 frames, producing `[8,768]`.
- IMU: each recording's corresponding centered 5-second window starts from its first IMU timestamp, based on the source's shared app clock (capture starts back-to-back, without hardware sync). The six limb quaternions are expressed relative to chest, then averaged in eight time bins to `[8,54]`. Empty bins, incomplete timestamps, duplicate `(time,slot)` rows, nonfinite values, and non-unit quaternions fail closed.
- Train IMU features: `[120,8,54]`, SHA-256 `99463ae3c5e790007b87b1a89ee8e1d5c2292914cba1b845f80f6a0066c7efad`.
- Development IMU features: `[24,8,54]`, SHA-256 `c8ba17a3b6f6f2991cb2a5a79a73a8901f05cd8b64824868f075adb7e756e725`.
- Train VideoPrism features: `[120,8,768]`, SHA-256 `7d59d96ce8139b6ad87d06eed7eebbb846c8d303ffc3dc239f136770559068fb`.
- Development VideoPrism features: `[24,8,768]`, SHA-256 `0b5af6f2f68c038106dd5f25c316183802f5455d7a6fafcfc7b5f205ef0c2d37`.
- IMU extraction read 1,455,025,035 bytes across 184 byte-range requests in 287.1 seconds. Video extraction read 4,982,218,869 bytes across 660 range requests in 2,203.9 seconds. VideoPrism ran on `TFRT_CPU_0`; no GPU training/inference was used.
- Timing estimate from the completed per-record logs (not repeated downloads of the same media): video extraction averaged 15.45 seconds per train video (120 records) and 14.60 seconds per development video (24 records), or 15.30 seconds per video over all 144. For a three-video preflight, budget about 46 seconds based on this run; individual records varied, so allow extra time for unusually large/slow clips. The corresponding IMU extraction averaged 1.45 seconds per train row and 2.83 seconds per development row. End-to-end video time was 36 minutes 44 seconds. These are observed-run estimates, not guarantees for a future network/session.

The checked-in feature code is in `scripts/eidon_range_io.py`,
`scripts/extract_eidon_sensor_features.py`, and
`scripts/extract_eidon_videoprism_features.py`. Run configs, media-derived
features, predictions, and checkpoints remain under `C:/CodexArtifacts/` and
are not in Git.

## Readout experiments

Both readouts use seed 17, the exact same 120 training queries and 24
development queries, channel-wise normalization fit on train recordings only,
AdamW at `1e-3`, weight decay `0.01`, gradient clipping at 1.0, and a
32-dimensional shared scoring space. The pooled readout averages each feature
sequence before projection; the LSTM readout uses separate unidirectional LSTMs
for video and sensor sequences. The latter has 120,193 trainable parameters;
the pooled head has 32,641. Including frozen VideoPrism, the measured
Video+IMU diagnostic has 114,485,377 / 114,397,825 parameters respectively.
This excludes any text, image, or audio towers and is not a complete Omni model.

| Run | Candidate | Completed epochs | Best epoch | Train accuracy at best | Development Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early-stop attempt | pooled | 21 | 1 | 25.8% | 29.2% | 1.3870 | 0.7504 | 0.0378 | 0.2539 |
| early-stop attempt | LSTM | 25 | 5 | 60.8% | 29.2% | 1.3842 | 0.7490 | 0.0389 | 0.2527 |
| fixed 2,000-epoch curve | pooled | 2,000 | 1 | 25.8% | 29.2% | 1.3870 | 0.7504 | 0.0378 | 0.2539 |
| fixed 2,000-epoch curve | LSTM | 2,000 | 5 | 60.8% | 29.2% | 1.3842 | 0.7490 | 0.0389 | 0.2527 |

Under a uniform target-index assumption, the population reference is expected
Accuracy 25%, NLL `ln(4)=1.3863`, Brier 0.75, ECE 0, and mean confidence 0.25.
This is not the empirical argmax score on the 24 examples: target positions
occurred 4/5/8/7 times, and deterministic `argmax` ties at option 0 (4/24 or
16.7%, empirical ECE 0.0833). The pooled head's NLL is slightly worse than the
uniform reference. The LSTM reduces NLL by 0.0021 on the same 24 examples, but
its paired group-bootstrap interval versus uniform is only a development
diagnostic and checkpoint-selection bias is not corrected.

### Fixed long learning curve

| Epoch | Pooled train Acc / NLL | Pooled dev Acc / NLL | LSTM train Acc / NLL | LSTM dev Acc / NLL |
|---:|---:|---:|---:|---:|
| 0 | 23.3% / 1.387 | 20.8% / 1.387 | 23.3% / 1.386 | 33.3% / 1.385 |
| 5 | 57.5% / 1.371 | 25.0% / 1.388 | 60.8% / 1.372 | 29.2% / 1.384 |
| 25 | 88.3% / 1.103 | 16.7% / 1.404 | 92.5% / 1.139 | 20.8% / 1.388 |
| 100 | 100% / 0.013 | 16.7% / 2.386 | 100% / 0.003 | 25.0% / 2.348 |
| 500 | 100% / 0.000 | 16.7% / 3.079 | 100% / 0.000 | 20.8% / 2.844 |
| 1,000 | 100% / 0.000 | 16.7% / 3.303 | 100% / 0.000 | 16.7% / 3.054 |
| 1,500 | 100% / 0.000 | 20.8% / 3.417 | 100% / 0.000 | 16.7% / 3.179 |
| 2,000 | 100% / 0.000 | 20.8% / 3.511 | 100% / 0.000 | 12.5% / 3.263 |

At 2,000 epochs, pooled / LSTM CPU fit time was 14.6 / 48.4 seconds. Best and
final checkpoints were saved separately and have distinct SHA-256 hashes. The
early-stop and long-budget runs use the same seed and data; the fixed curve was
restarted from the same seed-17 initialization so its first steps are
reproducible and its best checkpoint is not replaced by the final weights.

## Interpretation and next gate

- **Execution:** PASS for the frozen-feature and fixed 2,000-epoch readout
  diagnostic. Both modalities were present in the paired input; train/dev
  contributor and asset overlap was checked; hashes and predictions are saved;
  option-order equivariance passed; no sealed audit was read.
- **R2-M input-necessity/quality gate:** NOT PASSED. Accuracy is 29.2% on a
  24-query, 3-contributor development set. Paired input controls are too close
  to uniform or inconsistent to show that synchronized video and IMU both
  contribute useful information. The strict ``both-required`` claim remains
  unsupported.
- **Temporal hypothesis:** NOT SUPPORTED within this readout/budget. Reversing
  sensor time did not hurt; the LSTM overfit training while development NLL
  worsened after epoch 5. This is not a general statement about LSTMs, xLSTMs,
  or the data source.
- **Remaining uncertainty:** Only one seed, three held-out contributors, six
  development groups, one 5-second center window, one pretrained video encoder,
  and one quaternion-only sensor representation were tested. The sync clock is
  software-started rather than hardware-triggered. The match task is narrower
  than real decision support, and the development split has already been used
  for model selection.
- **Next:** Do not tune this candidate further against the same development
  set. If continuing R2-M, use a fresh, sufficiently large contributor-held-out
  development cohort; validate exact synchronization and task labels first;
  and define a consequential action/choice task that needs both streams. Keep
  audio, image, text, the 95–98% target, and the full-model parameter gate open.

The 44-video candidate sealed-audit set remains sealed and untouched. This
diagnostic does not change any legacy Teacher, corpus, checkpoint, branch merge,
or product selection.
