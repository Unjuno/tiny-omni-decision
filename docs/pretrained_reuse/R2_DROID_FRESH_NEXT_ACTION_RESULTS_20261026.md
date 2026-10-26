# DROID fresh-shard next-action results

## Decision

On this one-step DROID development task, the 114.37M VideoPrism feature path
does **not** establish an advantage over sensor-only proprioception. The first
train-only-selected readouts overfit the video features severely. A
train-only L2-grid extension reduced that gap, but video, sensor, and fused
validation results all remained around the training-prior baseline. This is
task-specific development evidence, not product-model quality or a blind
benchmark result.

## Frozen data and target

The experiment used nine previously unused, metadata-pinned DROID 1.4.0
official-train shards, seed 17, CC-BY-4.0. The fixed shard list is
`580, 1635, 1018, 717, 577, 1634, 487, 748, 1768`; the previously used 12
shards were excluded. Source/recording identity and normalized instruction
content were grouped before splitting. The resulting train and validation
parts contain 275 and 150 unique recording groups respectively, with zero
cross-split recording or normalized-instruction overlap. Three episodes were
too short for the predeclared 8-state early window and were excluded; all seven
target classes remain represented in train.

The input is three camera views and 14 raw proprioception values at eight
uniformly spaced steps from the first quarter of the episode. The target is
the dominant signed XYZ direction of commanded Cartesian velocity at the
immediately following step, with a `stop` class. It is a narrow robot
next-action task. The split comes from official train data and is nonblind.

| Split | Episodes | stop | x- | x+ | y- | y+ | z- | z+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 275 | 29 | 8 | 102 | 18 | 23 | 71 | 24 |
| Validation | 150 | 8 | 7 | 58 | 11 | 16 | 42 | 8 |

## Models and selection

The video encoder was frozen VideoPrism Base, revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
Each view is represented by eight temporally pooled 768-dimensional vectors;
the linear readout receives separate early-half and late-half means. Sensor
input is the raw `[8,14]` sequence. The compared L2 multinomial heads have
10,759 (video), 791 (sensor), and 11,543 (video+sensor) trainable parameters.
VideoPrism itself has 114,365,184 parameters and a 457,475,296-byte checkpoint;
the full encoder artifact exceeds the provisional 200 MB package limit.

Regularization is selected using only five-fold recording-group CV on train,
seed 17, macro NLL. One initial fit reached the legacy 500-iteration optimizer
cap in a video training fold; it produced no validation score. That failure is
preserved. Raising the L-BFGS-B iteration cap to 2,000 kept the objective and
folds unchanged and converged. In that baseline, all arms selected L2=100, the
largest value tried. Train-only CV macro NLL continued improving at that
boundary, which motivated a predeclared follow-up grid of `[100, 1000, 10000]`.
The follow-up selected 1,000 for all three arms. Validation was consequently
scored twice; all validation results below are nonblind development results.

## Validation results

The fixed initial L2 grid produced:

| Arm | Accuracy | Macro Acc. | NLL | Macro NLL | Brier | ECE-10 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Video | 34.00% | 20.56% | 1.9601 | 2.5888 | 0.8487 | 0.2099 | 0.5441 |
| Sensor | 41.33% | 18.41% | 1.5730 | 2.1181 | 0.7270 | 0.0528 | 0.4141 |
| Video + sensor | 37.33% | 18.63% | 1.8891 | 2.5200 | 0.8226 | 0.1718 | 0.5425 |
| Train-prior baseline | 38.67% | 14.29% | 1.6424 | 2.2542 | 0.7530 | 0.0158 | 0.3709 |

All three arms selected L2=100. The high video/fusion train Accuracy (98.55%
and 99.27%) fell to 34.00% and 37.33% on validation. Sensor train/validation
Accuracy was 51.64% / 41.33%. That is strong evidence of readout overfitting
for video features at this sample size.

The stronger-L2 follow-up produced:

| Arm | Accuracy | Macro Acc. | NLL | Macro NLL | Brier | ECE-10 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Video | 38.67% | 15.51% | 1.6272 | 2.1681 | 0.7517 | 0.0856 | 0.4083 |
| Sensor | 40.67% | 15.59% | 1.6083 | 2.1950 | 0.7409 | 0.0515 | 0.3689 |
| Video + sensor | 40.00% | 16.19% | 1.6065 | 2.1461 | 0.7427 | 0.0772 | 0.4088 |
| Train-prior baseline | 38.67% | 14.29% | 1.6424 | 2.2542 | 0.7530 | 0.0158 | 0.3709 |

The video/fusion train Accuracy fell to 63.27% / 64.73%; sensor train
Accuracy was 40.00%. Validation Accuracy stayed near the 38.67% train-prior
baseline. At this stronger regularization, fusion and sensor are nearly tied
in point metrics. The 5,000-replicate paired recording-group bootstrap
(seed 17) for fused minus sensor gave Accuracy difference 95% interval
`[-7.35, +6.00]` percentage points and NLL difference interval
`[-0.0724, +0.0711]`; it does not support a reliable fusion gain.

Minority labels remain weak. Under the stronger-L2 follow-up, per-class
validation `(count, Accuracy, NLL)` was:

| Class | Video | Sensor | Video + sensor |
|---|---|---|---|
| stop | (8, 0%, 1.918) | (8, 0%, 2.160) | (8, 0%, 1.891) |
| x- | (7, 0%, 3.334) | (7, 0%, 3.401) | (7, 0%, 3.280) |
| x+ | (58, 77.6%, 1.000) | (58, 94.8%, 0.984) | (58, 77.6%, 0.983) |
| y- | (11, 0%, 2.768) | (11, 0%, 2.694) | (11, 0%, 2.743) |
| y+ | (16, 0%, 2.350) | (16, 0%, 2.404) | (16, 0%, 2.339) |
| z- | (42, 31.0%, 1.438) | (42, 14.3%, 1.329) | (42, 35.7%, 1.411) |
| z+ | (8, 0%, 2.369) | (8, 0%, 2.392) | (8, 0%, 2.376) |

This class pattern and the low macro Accuracy show why aggregate Accuracy near
the prior is not sufficient. No arm is near the project’s 95–98% target.

## Compute and artifact integrity

- Video feature extraction: 1,452.45 seconds on `TFRT_CPU_0`; no GPU used.
- Readout evaluation: 47.40 seconds for the initial grid and 18.42 seconds for
  the stronger-L2 follow-up; SciPy CPU implementation.
- Environment: Windows, Python 3.11.16, NumPy 1.26.4, SciPy 1.17.1, JAX
  0.6.2, TensorFlow 2.18.1; CPU Intel Core i7-12700H. No cloud compute or
  compute fees.
- The RTX 3080 remained occupied by an unrelated process; it was not changed.
- Original 8.57 GB dataset remains outside Git. The final receipt was generated
  by re-verifying already-downloaded objects; its 9.997-second duration is not
  the original network-transfer duration. The first transfer duration was not
  captured because a post-download license-check bug stopped before receipt
  writing; the MD5-verified files were preserved and safely re-verified.
- Sealed audit: not loaded. Encoder weights: frozen. No product checkpoint,
  ternary model, adapter, or runtime artifact was created.

Important hashes:

| Artifact | SHA-256 |
|---|---|
| Selection manifest | `cec05f0e1a39de3d7021e74ec390c745a0e31ce9a3d0763344864dcb2a2e34aa` |
| Download receipt | `5d5a5ba5d59b8c76790db1c9399f6490e70098ad390efafe5ef7f94df63ee6f7` |
| Task manifest | `b7315c23bad4f9b195828c934cc3a87aeb34d36b1b1180a8e8152f783148e21d` |
| Feature summary | `4a764517ed01c0ac0b9727d53b0287734fa9e4325d0937ce58fb91f7edfb221f` |
| Video features | `a445bd638ca3500b64360215f1033f63e381e2aa38df9856b38dd6b912041585` |
| Sensor features | `de8dc6de9afec85932b31d08d94ed58d097896e03ad1098813fd9f2c7ce372b9` |
| Initial-grid evaluation report | `ed5c750c062b38a47b6dbc1f93af150b4eec426672024f1ae893f004000ef803` |
| Stronger-L2 evaluation report | `97c21fa27c8965adef82147961f124bc332c93da933145783b1379d8fa507a21` |
| Stronger-L2 validation predictions | `0c9ee9ba173192e06255cc08f00524d40f938dbc15fe554cfed75331184b7b38` |
| 500-iteration failure log | `7e5a3bb45b9ef9d438978ab836072bebbb48bb2702518ee66a5d60c2489699001` |

## Next implication

This result does not justify adding a fusion module or promoting a video+sensor
model. The linear video readout overfits under weak regularization, and the
stronger train-only penalty leaves all validation arms close to the class
prior. Further work should focus on better task/data coverage or a
predeclared, train-only representation/readout regularization study before
adding fusion capacity. This does not establish a general limitation of
pretrained video representations or of multimodal Decision models.
