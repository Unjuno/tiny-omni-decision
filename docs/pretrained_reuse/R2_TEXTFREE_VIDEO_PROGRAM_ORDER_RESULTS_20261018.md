# CLEVRER typed-program order probe results

**Status: completed; positional typed-program pooling is not adopted.** This
single-seed development experiment tests whether restoring order information to
the typed-program readout helps temporal descriptive video decisions. It does
not establish a general temporal-model ceiling or product quality.

## Run and provenance

- Source commit: `fa0bfd773f7bcd51ad5667e7e2984440ca50fca0`.
- Command: `python scripts/run_textfree_video_program_order_after_gpu_yield.py --config configs/pretrained_reuse/textfree_video_program_order_v1.json --output-dir C:\\CodexArtifacts\\pretrained-reuse-r2\\clevrer-program-order-fit-v1-attempt-02 --max-wait-seconds 300`.
- Attempt output: `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-program-order-fit-v1-attempt-02` (kept outside Git).
- Run status: `PASS_TEXTFREE_VIDEO_PROGRAM_ORDER_FIT`; started 2026-10-18 10:08:43.492 UTC, finished 10:10:00.389 UTC; 76.896 s wall time; local NVIDIA GeForce RTX 3080 Laptop GPU; cloud cost $0.
- Frozen train manifest: 512 rows / 256 scenes; SHA-256 `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`.
- New selection validation: 90 rows / 45 unseen scenes, exactly 45 temporal and 45 static descriptive; raw question SHA-256 `fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d`.
- Train/validation sample, scene, media, and normalized question/options overlap: zero. All selected validation video files were hash checked. The validation was used for checkpoint selection and is neither blind nor a sealed audit.
- Config SHA-256 `349f238fdafaecff264d91768eb39bc440abec43ed4fbcd60935aeab69dc2050`; saved effective config SHA-256 `f0866bfd5af600aa074daa8a974059b747aa9cc870cf27e04bfb2d975811eb4a`; input-preflight SHA-256 `3aca1f4334e48f5b1b12aa8afa7ff35af4e9e3a928d6f056647a11e2b42f8da3`.
- `run-report.json` SHA-256: `52b8014f6e97f91ebc500424473abb9c0b903db4352054a51ce9b41bc4b07a69`. Full per-example predictions, histories, checkpoints, and integrity receipts remain in the attempt directory; no model weights or media are committed.

The run waited for the RTX 3080 to become idle under the cooperative GPU-yield gate. The earlier attempt that failed before an optimizer update because deterministic CUBLAS configuration was late remains preserved separately. The successful runner sets that configuration before CUDA initialization. No existing training process was stopped or edited.

## Fixed comparison

Both arms used seed 17, the same frozen VideoPrism-B features, BiLSTM frame block, option scorer, 512-row train set, row order, AdamW (`lr=0.001`, `weight_decay=0.0001`), batch 64, gradient clip 1, 256 epochs, and 2,048 updates. There were 8 updates per epoch. The only difference was mean pooling of typed-program embeddings versus mean pooling after adding fixed sinusoidal positions. Position vectors add zero trainable parameters. Both arms have 227,377 trainable parameters and 114,592,561 isolated video-path parameters before quantization.

### Fixed-step curves

Validation macro metrics average the two question types, with 45 examples each. ECE uses 15 bins. `selector NLL` is the pre-existing macro question-type selector score. Train statistics cover the fixed 512 training rows, so they are a training-fit diagnostic, not a generalization-gap estimate.

| Arm | Updates | Train CE | Train Accuracy | Val Accuracy | Val NLL | Val Brier | Val ECE | Selector NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean baseline | 512 | 0.1320 | 0.9863 | 0.5444 | 1.4103 | 0.6337 | 0.2507 | 1.4418 |
| Mean baseline | 1,024 | 0.0013 | 1.0000 | 0.5444 | 2.2261 | 0.7535 | 0.3701 | 2.2483 |
| Mean baseline | 1,536 | 0.0004 | 1.0000 | 0.5556 | 2.4764 | 0.7762 | 0.3610 | 2.4764 |
| Mean baseline | 1,920 | 0.0002 | 1.0000 | 0.5556 | 2.5949 | 0.7865 | 0.3806 | 2.5861 |
| Mean baseline | 2,048 | 0.0002 | 1.0000 | 0.5444 | 2.6260 | 0.7887 | 0.3706 | 2.6150 |
| Positional mean | 512 | 0.1550 | 0.9688 | 0.5111 | 1.3072 | 0.6816 | 0.2598 | 1.3609 |
| Positional mean | 1,024 | 0.0014 | 1.0000 | 0.5000 | 2.2379 | 0.8435 | 0.3987 | 2.2982 |
| Positional mean | 1,536 | 0.0004 | 1.0000 | 0.5111 | 2.4686 | 0.8505 | 0.4251 | 2.5166 |
| Positional mean | 1,920 | 0.0002 | 1.0000 | 0.5111 | 2.5802 | 0.8546 | 0.4352 | 2.6243 |
| Positional mean | 2,048 | 0.0002 | 1.0000 | 0.5000 | 2.6089 | 0.8573 | 0.4357 | 2.6509 |

Both arms fit the training rows almost perfectly and validation NLL/calibration deteriorated after the early region. The fixed selector chose mean baseline epoch 33 / update 264 and positional mean epoch 21 / update 168. The selected checkpoints were exactly reload-verified (maximum logit difference zero). Selection is reported separately from fixed-step endpoint metrics.

### Selected validation checkpoints

| Arm | Selected step | Type | n | Accuracy | NLL | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|
| Mean baseline | 264 | Temporal descriptive | 45 | 0.5778 | 0.9640 | 0.5448 | 0.2332 |
| Mean baseline | 264 | Static descriptive | 45 | 0.6222 | 1.1252 | 0.5571 | 0.1403 |
| Mean baseline | 264 | Macro across types | 90 | 0.6000 | 1.0446 | 0.5510 | 0.1867 |
| Positional mean | 168 | Temporal descriptive | 45 | 0.5556 | 0.9518 | 0.5351 | 0.0612 |
| Positional mean | 168 | Static descriptive | 45 | 0.5333 | 1.1123 | 0.5922 | 0.1360 |
| Positional mean | 168 | Macro across types | 90 | 0.5444 | 1.0321 | 0.5637 | 0.0986 |

The candidate's macro selector score was 1.1274 versus 1.1119 for baseline (higher is worse). At selected checkpoints, positional pooling changed temporal NLL by only -0.0121 and Brier by -0.0097 while temporal accuracy fell 2.22 percentage points. Static accuracy fell 8.89 points and static Brier worsened 0.0351. Macro accuracy fell 5.56 points; macro NLL improved 0.0125, macro Brier worsened 0.0127, and macro ECE improved 0.0881.

### Paired uncertainty and resources

Scene-cluster paired bootstrap, candidate minus baseline, 5,000 repetitions,
seed 17, percentile 95% intervals:

| Type | Metric | Difference | 95% interval |
|---|---|---:|---:|
| Temporal | Accuracy | -0.0222 | [-0.1556, 0.1333] |
| Temporal | NLL | -0.0121 | [-0.1744, 0.1356] |
| Temporal | Brier | -0.0097 | [-0.1035, 0.0802] |
| Static | Accuracy | -0.0889 | [-0.2222, 0.0444] |
| Static | NLL | -0.0129 | [-0.2188, 0.1797] |
| Static | Brier | +0.0351 | [-0.0568, 0.1247] |

Every interval includes zero. Each arm took 16.28 / 19.66 seconds of training
and 18.55 / 20.31 seconds of evaluation and serialization, respectively;
peak CUDA allocation was 158,029,312 bytes for each. Total sequential wall
time was 76.90 seconds. Both final checkpoints are retained separately from
their best checkpoints. Full per-example outputs and history hashes are in the
external attempt record.

## Interpretation and next action

This result does **not support adopting fixed positional mean pooling**: its
selector score and macro accuracy were worse, and the paired intervals are
wide. The tiny temporal NLL/Brier changes are inconclusive. This is also not
evidence that order is irrelevant or that the model family has reached a
ceiling: validation contains only 45 scenes and two descriptive question types,
and is reused for checkpoint selection. Strong late overfitting means this
training setup cannot distinguish insufficient temporal representation from
task coverage, label/readout limits, or regularization. No sealed audit was
loaded; no quantization, product promotion, additional seed, or budget
extension was performed. Existing v1/v2 artifacts and source corpora were not
modified.

Before another fit, prioritize a broader, semantically representative temporal
task set and a fresh scene-grouped development split, then preregister a
regularized learning-curve comparison. Do not rerun these two arms on the same
selection validation or treat this small probe as independent evidence.
