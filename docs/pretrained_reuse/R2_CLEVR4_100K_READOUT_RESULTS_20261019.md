# Clevr-4 100k frozen DINOv2 readout results (2026-10-19)

## What ran

Using only the official Clevr-4 `train` partition, seed 17 created a fixed
72,010-image train split and an 8,002-image internal validation split by asset
identity. Every selected PNG was SHA-256 hashed. Train/validation media
overlap was zero, and neither split overlapped any of the 8,424 images in the
prior Clevr-4 training manifest. The archive's separate public `val` partition
was not decoded or opened. This internal validation set is development data,
not a sealed audit.

The frozen input is DINOv2-small CLS plus mean-patch features (768 dimensions),
with frozen MiniLM option text (384 dimensions). A 147,713-parameter existing
candidate scorer learns to rank the ten supplied classes for each of four
taxonomies. Only this scorer is trainable. The encoder-plus-readout inventory
is 39,594,113 parameters, not a complete standalone multimodal product.

Two fits ran on the exact same features and split. Attempt 01 used seed 17,
eight epochs, batch 64 images (four decisions per image per update), AdamW at
0.001 / 0.0001, and gradient clipping at norm 5. Attempt 02 corrected the
optimizer mismatch: seed 17, eight epochs, batch 64 individual image-taxonomy
decisions, the same AdamW settings, and no clipping. Both selected the earliest
minimum macro validation NLL and ran the complete schedule; epoch 8 was
selected in both. The prior 10k-trained readout was evaluated on the same
validation IDs before either fit.

## Learning curve

Training Accuracy below is an online minibatch metric while weights are
changing; it is not a fixed train-set evaluation and must not be read as a
generalization gap.

| Epoch | Online train CE | Online train Accuracy | Validation macro Accuracy | Macro NLL | Macro Brier | Macro ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6728 | 77.31% | 88.45% | 0.3098 | 0.1570 | 0.0464 |
| 2 | 0.2808 | 88.93% | 89.46% | 0.2580 | 0.1398 | 0.0174 |
| 3 | 0.2462 | 90.01% | 89.65% | 0.2494 | 0.1362 | 0.0109 |
| 4 | 0.2329 | 90.44% | 90.31% | 0.2435 | 0.1297 | 0.0106 |
| 5 | 0.2233 | 90.82% | 89.40% | 0.2535 | 0.1395 | 0.0175 |
| 6 | 0.2168 | 91.03% | 90.97% | 0.2259 | 0.1223 | 0.0074 |
| 7 | 0.2125 | 91.21% | 90.46% | 0.2296 | 0.1266 | 0.0109 |
| 8 | 0.2086 | 91.31% | 91.04% | 0.2163 | 0.1202 | 0.0058 |

The matched attempt used individual questions as updates:

| Epoch | Online train CE | Online train Accuracy | Validation macro Accuracy | Macro NLL | Macro Brier | Macro ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6075 | 77.48% | 84.55% | 0.3841 | 0.1954 | 0.0157 |
| 2 | 0.3613 | 85.13% | 85.99% | 0.3355 | 0.1740 | 0.0190 |
| 3 | 0.3171 | 87.02% | 87.91% | 0.2969 | 0.1565 | 0.0082 |
| 4 | 0.2928 | 88.06% | 88.62% | 0.2893 | 0.1498 | 0.0097 |
| 5 | 0.2753 | 88.83% | 88.73% | 0.2830 | 0.1483 | 0.0130 |
| 6 | 0.2625 | 89.32% | 89.19% | 0.2623 | 0.1416 | 0.0150 |
| 7 | 0.2532 | 89.71% | 89.74% | 0.2521 | 0.1362 | 0.0119 |
| 8 | 0.2460 | 90.05% | 90.10% | 0.2478 | 0.1331 | 0.0120 |

## Same-validation comparison

All values use 8,002 images per taxonomy (32,008 option decisions). Both
readouts use the same frozen DINOv2 and MiniLM representations, candidate
ordering, preprocessing, and evaluation code.

| Readout | Procedure | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---|---:|---:|---:|---:|---:|
| Prior 10k-train readout | batch 64 decisions, no clipping | 84.30% | 0.4021 | 0.2048 | 0.0185 | 84.75% |
| 100k attempt 01 | batch 64 images / 256 decisions, clip 5 | 91.04% | 0.2163 | 0.1202 | 0.0058 | 91.34% |
| 100k attempt 02 | batch 64 decisions, no clipping | 90.10% | 0.2478 | 0.1331 | 0.0120 | 90.95% |

The corrected, optimizer-matched attempt improves macro Accuracy by 5.80
percentage points, NLL by 0.1543, and Brier by 0.0717 versus the prior readout
on the same validation IDs. Attempt 01 is retained but is not used as the
matched comparison because of its batch/clipping differences.

| Taxonomy | Prior Accuracy | Attempt 01 Accuracy | Matched Accuracy | Matched NLL | Matched Brier | Matched ECE |
|---|---:|---:|---:|---:|---:|---:|
| Color | 96.33% | 99.36% | 98.80% | 0.0370 | 0.0189 | 0.0022 |
| Texture | 88.04% | 95.34% | 94.28% | 0.1693 | 0.0843 | 0.0129 |
| Count | 53.52% | 69.61% | 67.48% | 0.7784 | 0.4264 | 0.0313 |
| Shape | 99.33% | 99.85% | 99.83% | 0.0067 | 0.0028 | 0.0014 |

The matched result supports a useful data-coverage effect for this synthetic
benchmark/readout pairing. Count remains the bottleneck at 67.48%; neither
attempt meets a robust all-task image floor, much less the requested 95–98%
four-modality product goal. Attempt 01's higher validation score is not a
reason to prefer its mismatched optimizer procedure. The validation set was
already exposed before the corrective run; neither score is blind evidence.
In the matched run, validation accuracy and NLL improved across the eight
epochs overall; this provides no obvious learning-curve overfit signal, but
does not establish absence of overfitting beyond this split.

## Run integrity and artifacts

- Source commit: `8336a98316267cfc7881ee330df8e8f919d2479e`.
- Archive SHA-512:
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
- Official split counts: train 80,012; public val 19,988; no test partition.
- Unique internal train/validation assets: 72,010 / 8,002; all media hashes
  unique; all listed split/prior-manifest media overlaps are zero.
- Feature extraction used the RTX 3080 Laptop through HotUGPU: 3,200 MiB
  cooperative reservation, 13,015 MiB free at child start, 330,529,280-byte
  peak allocation, 921.68 seconds. Selected image files were not extracted.
- Readout fit and validation: 25.66 seconds, 109,306,368-byte peak allocation.
- Corrective matched readout fit and validation: 64.84 seconds, 110,012,416-byte
  peak allocation; 36,008 optimizer updates.
- Execution caveat: the runner applies gradient clipping at norm 5.0, while
  the pre-run protocol did not state a clipping rule and the previous runner
  did not clip. This is a small unplanned optimizer difference. The code and
  run receipt record it; consequently the comparison is matched on data IDs,
  representation, model and evaluation, but not an exact optimizer recipe.
- Attempt 01 consumed 576,080 image presentations in 9,008 updates; attempt 02
  consumed 2,304,320 image presentations in 36,008 updates. Both presented
  2,304,320 image-taxonomy decisions across eight epochs.
- The selected checkpoint reload produced exactly the same 32,008 prediction
  rows and metrics as the saved best-epoch predictions.
- The selected 100k train/validation ID-order hashes are
  `cc0c59c90ab363e46abfabd5910f7eab06d79c418da52650d741436f3df0d96c` and
  `58d3479904f820f00bdb374a33df759782985d651058a06123d204edd59c0d05`.
- Index report SHA-256 is
  `def60b282d0693ee52a2872bfe5953a2a9b1427a82f0d4af54341c57916e1724`; the
  hashed-media report SHA-256 is
  `f7847e2bfe733d4e21000a76786988c7bbd3ed639eaf812ab301291e57a861e5`.
- Feature runner SHA-256 is
  `1bec1bf47b3d30b980777dd54d4f5d26efc66474b6c8feddf25792b40a638f27`;
  feature result SHA-256 is
  `9e0ec7f791f72bb4c83c80c0131327bae17ca8b8153433cd522bf48c6c2c9d1f`.
  Train/validation feature arrays hash to
  `2480b0ea486a55efe599337eefda38b587e5501bc087aca96c1f5db3da00b622` and
  `70b0b5875930fa82a4ced699df2ae9c490f952a8e730242acae6ac7d90ab9fc5`.
- Attempt 01 config/history/result SHA-256 values are
  `836340784fe928efcf338851d7624e2fb12bbdb6ce65eff6fe48409f85bfc63a`,
  `5b1ab12bae579bef7dbfec5422f75cee3138e38cd68ea5193e627a63a84e89ba`, and
  `93dcb699843c05b2e89ade369b9cc864d7786d391e53626603bc4bc02f90f939`.
  Attempt 02 config/history/result SHA-256 values are
  `0cf2ca4acc082afea7cd3ceb8b3fc73f90130055c3f0a66ad877216a49cf7c98`,
  `3afee97a50bf5e790484dd5137c14e33135f9b47a50b2459b8fb3871c56e8bd9`, and
  `e087811e6c783601f9a0afaca0ab510b5c12e50eb7faf52fd9730792e347fa69`.
- Attempt 01 and matched attempt 02 training-runner SHA-256 values are
  `aa814bc424862c329d2a6491367b47c3b589c29ff0c0727ab55d5201a6400216` and
  `b8a0ca2ac9988809e309cda85bc053d8aaddbfe96a4b5bc8db80e7b6ce2b828c`.
- No quantization, cloud compute, model promotion, or official-val evaluation.
  Local RTX 3080 GPU compute cost was $0; no paid/cloud compute was used.

Attempt 01 artifacts are under
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-100k-readout-seed17-v1\`;
the corrected matched run is under
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-100k-readout-seed17-matched-attempt-02\`.
The hashed corpus index and frozen features are in adjacent
`clevr4-100k-index-seed17-v0`,
`clevr4-100k-index-seed17-media-hashed-v1`, and
`dinov2-clevr4-100k-seed17-features-v1` directories. The best scorer SHA-256
is `5cabe9ce697cdfa84ab90b27b7127f0d72462b5c54915dc3b1c807178bd2b0b8`;
the attempt-01 final scorer SHA-256 is
`0413f5cd73fc926e2ccc48cc503c81d2bac44a8ff5bf1fbb231f5765c952b`. The
matched attempt-02 best/final SHA-256 values are
`6f08e88014ca043570888264e857821eb71b13106d9a3673ea81240a192a0439` and
`62ee42ab7cff339ec2b240acc7e4effd4a936d34b82369a3d191e9ca55380be6`.
The readout weights, media and features are intentionally not in Git.

## Interpretation and next step

The optimizer-matched data expansion improved all four narrow Clevr-4
attribute tasks relative to the prior readout on one identical fresh validation
split. The matched accuracy gain is largest for count (+13.96 points), which
is still the bottleneck. This supports more unique examples for this specific
dataset/readout combination, but the development split has now been used for
both checkpoint selection and post-hoc comparison. It does not prove
generalization to natural images or justify expanding this synthetic benchmark
indefinitely.

Keep this as an image-path research result. Next, test whether count errors are
driven by image representation or scorer limitations using a small, predeclared
diagnostic on the same frozen features, or move to an independent natural-image
decision task with its own untouched split. Do not use the public official
validation partition for checkpoint tuning and do not infer multimodal quality
from this image-only result.
