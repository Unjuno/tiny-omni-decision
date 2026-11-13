# VideoMAE Clevr-4 spatial-grid candidate — results (2026-11-13)

## Result

One predeclared 4×4 spatial-pooling candidate completed on the frozen train and
selector partitions. The best checkpoint was selected by equal-taxonomy
selector NLL (epoch 8). On the same 1,000-image-per-taxonomy development
selector used by the global-vector baseline, macro Accuracy improved from
79.625% to 83.525%, macro NLL from 0.50820 to 0.39678, macro Brier from
0.24946 to 0.20748, and ECE-15 from 0.02765 to 0.01829. Count Accuracy rose
from 42.6% to 51.8%; it remains the weakest task. This selector has been used
for model selection and is not an independent evaluation. The separate 5,296
image reporting partition and sealed audit were not extracted or loaded for
this candidate.

| Taxonomy | Selector n | Global Accuracy / NLL / Brier / ECE-15 | Spatial Accuracy / NLL / Brier / ECE-15 |
|---|---:|---:|---:|
| Color | 1,000 | 99.30% / 0.03239 / 0.01370 / 0.01735 | 99.30% / 0.01955 / 0.01057 / 0.00413 |
| Texture | 1,000 | 77.30% / 0.63508 / 0.30843 / 0.04337 | 83.60% / 0.45647 / 0.23240 / 0.02506 |
| Count | 1,000 | 42.60% / 1.34047 / 0.66482 / 0.03539 | 51.80% / 1.09397 / 0.57767 / 0.03842 |
| Shape | 1,000 | 99.30% / 0.02487 / 0.01088 / 0.01451 | 99.40% / 0.01713 / 0.00927 / 0.00554 |
| **Equal-taxonomy macro** | **4,000 decisions** | **79.625% / 0.50820 / 0.24946 / 0.02765** | **83.525% / 0.39678 / 0.20748 / 0.01829** |

The count ECE is slightly worse (+0.00303) despite gains in count Accuracy,
NLL, and Brier. This is a task-specific development-selector result, not
evidence of general image reasoning or the project's 95–98% multimodal target.
Both candidates were evaluated repeatedly on this selector during their
respective eight-epoch fits; the matched comparison is useful for screening,
not a blind estimate of gain. The prior one-time reporting metrics from the
global-vector run are not compared here because this candidate deliberately
did not access that partition.

## Fixed setup and execution

- Source commit: `4066603df62d0f770e16bdf2e7b454a5b283f87d`.
- Config: `configs/pretrained_reuse/path_a_image_videomae_clevr4_spatial_grid_4x4_seed17.json`; SHA-256 `f4f2658be5fb6e614a358d3f3ed8773a4fc4718a641f07e6ac39db63746fda27`.
- Seed 17, VideoMAE V2 ViT-Small K710 frozen encoder, 4×4 adaptive-pooled patch grid, hidden size 128, taxonomy embedding size 16, eight epochs, batch 512 task examples, AdamW LR `1e-3`, weight decay `1e-4`, constant scheduler. No sweep, extra seed, epoch extension, or changed data policy.
- Training inputs: 72,010 train images and 1,000 selector images, all unique assets; train-selector ID and media overlap are zero. Feature rows: 73,010. Reporting and sealed partitions are absent from the derived input manifest.
- Each train image contributes four taxonomy decisions per epoch: 288,040 task examples per epoch and 2,304,320 task-example presentations across the fixed eight epochs (72,010 unique train image assets).
- The encoder averages final tokens over temporal tubelets before 4×4 spatial pooling. The source images are repeated static frames, so no temporal information is tested.
- The readout has 789,962 trainable parameters; combined encoder plus readout is 22,943,248 parameters. The best checkpoint was reloaded and reproduced selector NLL exactly. Best epoch and final epoch are both 8.

Feature extraction ran from `2026-11-13T06:16:57.678871Z` to
`2026-11-13T07:27:19.748866Z`, 4,227.46 seconds (70.46 minutes), on the local
NVIDIA RTX 3080 Laptop GPU, FP32, batch 8, Python 3.11.9, PyTorch 2.6.0+cu124.
Peak CUDA allocation was 1,235,062,784 bytes. The 1,794,293,888-byte feature
array has shape `[73010, 6144]`, is finite, and has SHA-256
`0ad2525a73c95cc6aae7ab4ea6677362a7b1bda15181c98409dc36d2f81d9424`. The
feature manifest SHA-256 is
`4dab19d558937a88a17d9852d34a238142b7c36a9e0ac20ea10cc5609af76603`.

The readout fit took 29.58 seconds and peaked at 139,059,200 allocated CUDA
bytes. No paid compute was used. No encoder, source data, prior artifact, or
source file was modified.

## Learning curve

Training accuracy and CE are rolling metrics over shuffled batches, not fixed
training-set evaluations or a train-validation generalization gap.

| Epoch | Train rolling accuracy | Train CE | Selector macro Accuracy | NLL | Brier | ECE-15 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6064 | 1.1244 | 0.7560 | 0.6442 | 0.3061 | 0.0536 |
| 2 | 0.7939 | 0.5363 | 0.7885 | 0.5267 | 0.2640 | 0.0308 |
| 3 | 0.8221 | 0.4487 | 0.8050 | 0.4734 | 0.2417 | 0.0228 |
| 4 | 0.8351 | 0.4099 | 0.8255 | 0.4496 | 0.2275 | 0.0192 |
| 5 | 0.8433 | 0.3856 | 0.8215 | 0.4273 | 0.2194 | 0.0226 |
| 6 | 0.8498 | 0.3670 | 0.8200 | 0.4288 | 0.2189 | 0.0285 |
| 7 | 0.8542 | 0.3548 | 0.8348 | 0.4124 | 0.2125 | 0.0154 |
| 8 (selected) | 0.8598 | 0.3410 | 0.8353 | 0.3968 | 0.2075 | 0.0183 |

Selector Accuracy and NLL do not improve monotonically at every epoch, while
the selected endpoint is the best NLL among the eight recorded epochs. This
single short curve does not establish an overfitting plateau or broader
generalization.

## Artifacts and integrity

External output directory:
`D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-spatial-readout-20261113-attempt-01`.
Its `run-report.json` SHA-256 is
`da7da76d0f43ef72c76d0440cc4f9c28f39628e58872ecdf3b9c157707e6a2db`.
Other recorded hashes:

- Frozen train/selector input receipt: `f93c925ad98cd907c5d395a83c6ec4b1b8136f4ee096c4fd241e5d10f483ba39`.
- Best readout checkpoint: `10e5f02524f7ae7dd6576950cb896c61fdcb71b270bacc923b82441f85cfc5bf`.
- Final readout checkpoint: `7f938adb9471e5e1b8b80181271cc18aab033a78851ef24848643115642d614f`.
- Training history: `1632eee2efe73488fb64671318d84862ec7d2950373568a13ef858464a960cba`.
- Selector predictions: `6ceee795fbda989a7903d90b7cc7a880e0aaefee74e952744badcb16ce93ddf2`.

The original global-vector feature/readout and the one-time reporting result are
unchanged. No sealed audit was accessed. This candidate is not promoted as a
product model. The next justified step is a fresh, task-relevant evaluation
gate; do not tune further on these selector IDs.
