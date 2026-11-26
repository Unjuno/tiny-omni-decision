# iNaturalist CC0 fresh-observer readout (2026-10-29)

## Scope and outcome

This is one seed-17 CPU experiment on an eight-species iNaturalist image
classification task. It is not an open-vocabulary Decision benchmark, a
temporal-video result, a multimodal-fusion test, or evidence that the
100–200M-parameter product target reaches 95–98% across modalities. The frozen
VideoPrism-B encoder has 114,365,184 parameters; only the 6,152-parameter
affine readout was trained. Still images were repeated across eight VideoPrism
frames, so this experiment does not measure temporal understanding.

The expanded readout improved over the existing readout on the same fresh
validation IDs. The candidate checkpoint was selected by validation NLL on
those IDs, so this is a development result with selection effects, not a blind
generalization estimate. It remains well below the 95–98% aspiration.

## Split audit and correction

The source is the live iNaturalist API, queried for Research Grade observations
with explicit CC0 observation and photo licenses. The source is not a dated
dataset snapshot. Candidate rows were limited to one photo per observer and
taxon and assigned with the existing seed-17 observer hash.

The first fresh-validation construction mistakenly treated the pages 1–10
validation rows as unused. A streaming media check detected an exact photo and
observation match with the already-scored observer-transfer validation cohort.
The run failed closed after 658 rows (654 train features and four validation
features); it produced no quality metrics. The colliding row and all 176
previously exposed observer-transfer validation identities were then added to
the excluded identity set. No previous media, features, checkpoint, or report
was modified.

The final split uses the previously unused pages 1–10 **train** observer groups
as validation (430 rows). These rows were metadata-only before this experiment:
they were never used to fit a readout or scored. The prior 176 exposed pages
1–10 validation rows are excluded. Pages 11–30 validation rows are also
excluded because they were used for selection in the earlier probe. Training
uses the retained train rows from pages 11–40: 921 existing rows plus 224 new
rows. The resulting fixed cohort has 1,145 train and 430 validation examples,
with every one of the eight taxa represented in both splits. Northern
Cardinal is the smallest class (76 train, 16 validation).

| Taxon ID | Train | Validation |
|---:|---:|---:|
| 3017 | 190 | 51 |
| 4956 | 145 | 49 |
| 6930 | 106 | 53 |
| 7089 | 133 | 54 |
| 9083 | 76 | 16 |
| 12727 | 121 | 53 |
| 13858 | 199 | 74 |
| 144455 | 175 | 80 |
| **Total** | **1,145** | **430** |

Observer, observation, and photo identity overlap is zero across the final
train/validation split. Exact media hash overlap is zero, as is pHash Hamming
distance ≤4. The selected new media was also checked against the legacy
train/validation media, the exposed observer-transfer media, and the complete
pages 11–30 feature receipt. The sealed audit was not accessed.

## Training protocol

- Source repository revision before the URL handling fix: `40428a767bfc49f51afa9734394360d179f57713`.
- VideoPrism-B revision: `be719a406d563b66f0ac969e7c94bab8e997c81a`.
- VideoPrism source revision: `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- Checkpoint SHA-256: `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Frozen feature pipeline: same eight repeated frames, 288-pixel short-side
  resize and center crop, mean pooled to 768 float32 features.
- Classifier: one affine 768-to-8 layer; 6,152 trainable parameters. Encoder
  frozen.
- Seed: 17; AdamW, learning rate `0.03`, weight decay `0.0001`, 200 epoch cap,
  early stop after 25 epochs without validation NLL improvement (`min_delta`
  `0.0001`). Train-split mean and standard deviation only.
- CPU fit: 2.25 seconds, 35 epochs run; best checkpoint selected at epoch 10.
- No additional seed or hyperparameter search was run.

The media downloader originally rejected uppercase `.JPG` suffixes despite the
same approved HTTPS iNaturalist host, matching photo ID, and square-photo path.
The path check now accepts case-insensitive `jpg`, `jpeg`, and `png` suffixes
while preserving the exact URL path and MIME validation. Three focused tests
pass. The stricter host, ID, redirect, size, license, decode, and hash checks
remain active.

## Learning curve and metrics

| Epoch | Train accuracy | Train NLL | Validation accuracy | Macro validation accuracy | Validation NLL | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.95% | 1.3303 | 54.65% | 55.63% | 1.5827 | 0.6722 | 0.2500 |
| 4 | 77.90% | 0.8732 | 71.63% | 66.58% | 1.2361 | 0.4490 | 0.1651 |
| 10 (selected by NLL) | 90.83% | 0.3655 | 74.88% | 71.90% | 1.1258 | 0.4004 | 0.1562 |
| 20 | 97.99% | 0.0814 | 76.05% | 75.08% | 1.3819 | 0.4031 | 0.1643 |
| 35 (stop) | 100.00% | 0.0144 | 75.81% | 74.71% | 1.4524 | 0.4132 | 0.1781 |

At the selected checkpoint, mean confidence was 90.50%. The train/validation
accuracy gap was 15.95 points, macro accuracy gap 17.16 points, and NLL gap
0.7603. Training loss continued down while validation NLL and calibration
worsened after epoch 10. This is direct evidence of overfitting for this
readout and cohort; epoch 20's higher accuracy was not used to override the
predeclared NLL selector.

## Matched baseline on the same validation IDs

The existing pages 11–30 readout checkpoint was evaluated once on the same
430 fresh validation IDs, with its own saved training normalization. No prior
validation scores are mixed into this comparison.

| Readout | Accuracy | Macro accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Existing readout | 68.84% | 62.91% | 1.3501 | 0.4825 | 0.1851 | 87.35% |
| Expanded readout, selected epoch 10 | 74.88% | 71.90% | 1.1258 | 0.4004 | 0.1562 | 90.50% |
| Change | +6.05 pp | +8.99 pp | −0.2243 | −0.0822 | −0.0289 | +3.15 pp |

Per-taxon results on these same IDs:

| Taxon ID | N | Existing accuracy | Expanded accuracy | Expanded NLL |
|---:|---:|---:|---:|---:|
| 3017 | 51 | 82.35% | 78.43% | 0.8202 |
| 4956 | 49 | 53.06% | 71.43% | 0.9558 |
| 6930 | 53 | 67.92% | 69.81% | 2.4973 |
| 7089 | 54 | 77.78% | 81.48% | 1.1438 |
| 9083 | 16 | 18.75% | 50.00% | 1.7259 |
| 12727 | 53 | 39.62% | 60.38% | 1.5954 |
| 13858 | 74 | 83.78% | 82.43% | 0.5362 |
| 144455 | 80 | 80.00% | 81.25% | 0.6184 |

Several classes improved substantially, but Mallard (3017) and House Sparrow
(13858) regressed slightly. Canada Goose (6930) remains poorly calibrated,
with validation NLL 2.4973 despite 69.81% accuracy. The small Cardinal sample
makes its point estimate especially uncertain. This is not evidence of
general-purpose decision quality.

## Artifacts and reproduction

All media, feature caches, checkpoints, predictions, and run metadata remain
outside Git under `C:\CodexArtifacts\`.

- Final metadata: `inat-birds-cc0-expansion-freshval-page1-10-train-heldout-20261029-attempt-02`.
  Train manifest SHA-256 `7bcfee86572230c6bb9f920b299194f436a82b982dae076800761b790948ffb2`;
  validation SHA-256 `681710df465ba25fe7e6b6991a239f348750623fab59c1890f8409ed85ed78c6`.
- Assembled features: `inat-birds-cc0-expansion-freshval-v2-merged-features-20261029-attempt-01`.
  Feature SHA-256 `f19fb03b8ee6015b20a8e153a38c7bdce3e008e3761b2de4953f7bc2c0cf1913`;
  feature-row SHA-256 `5550167e42b076c71720d9d0d4abb738ba5af0edfc8eed4b2c26cea0dd496ae3`;
  media receipt SHA-256 `715032453eb647be6837f0cefbf4711c6a6001d727f95b1200a2c874c30e6c11`.
- Readout run: `inat-birds-cc0-expansion-videoprism-linear-readout-freshval-20261029-attempt-01`.
  Config SHA-256 `a9c183d39b8fc31a3a2b9e357349c2f577481a9f578cae5b0fc2fbedae04fbb5`;
  selected checkpoint SHA-256 `089a64d670ab04b5cd5befc9f7e4897fd4f033aef1a29ea86be398c0f1a9b99d`;
  history SHA-256 `4570d376e3ee244c5647821b29a31493ccb3568d62d144c2b498d21cafffb0d5`;
  validation predictions SHA-256 `9266ffecfeef7795df38a37dd8cab9cfc89900b54e3beca69b4a28728ba32173`.
- Same-ID comparison JSON SHA-256 `2b5e046e78ad4f582bd686c93a9ac3329861d9b0dfa25d2ae6c904694ef097ee`;
  paired predictions SHA-256 `1f6937b6538fc2d09906bb2bd2671f15aeb1fe88db5b2802d539b4ee22a57025`.
- Failed initial extraction attempt is retained at
  `inat-birds-cc0-expansion-freshval-incremental-features-20261029-attempt-01`;
  it stopped on the exact exposed-validation media collision and was not used
  to train or score. Its verified 654-row prefix is reused only in the
  assembled cache; the four already-exposed validation features are excluded.

The sealed audit, other modalities, temporal reasoning, quantization, and
product promotion were not run. A clean 95–98% result was not reached. The
result supports only a modest held-out-observer improvement for this narrow
image task; the learning curve shows that more epochs alone overfit. A future
quality claim needs a fresh, larger, class-balanced source or a different
decision-relevant task, plus validation that is not used for checkpoint
selection before any blind audit.
