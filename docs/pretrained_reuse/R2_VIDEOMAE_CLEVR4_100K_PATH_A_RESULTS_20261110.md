# VideoMAE Clevr-4 100k Path A — measured results

## Outcome and claim boundary

The fixed Path A experiment completed feature extraction and all eight readout
epochs. The chosen checkpoint was epoch 8 by the preregistered minimum
equal-taxonomy NLL rule. The frozen encoder plus a small option scorer reached
79.63% macro Accuracy on the 1,000-image selector and 80.01% on the disjoint
5,296-image one-time internal reporting pool. This is evidence for a narrow,
closed-set synthetic image taxonomy task only. It is not a result for natural
images, free-form questions, audio, temporal video, sensor fusion, or the
product-wide accuracy target. No model is promoted.

## Fixed inputs and run provenance

- Source commit: `01d37d3c0350cc254407542ab1f10f7373d4d8ca`.
- Protocol/config: `docs/pretrained_reuse/R2_VIDEOMAE_CLEVR4_100K_PATH_A_PROTOCOL_20261110.md` and
  `configs/pretrained_reuse/path_a_image_videomae_clevr4_100k_seed17.json`.
- Effective config SHA-256: `97f32fd4a2172526c67265f7cd74a153c993fd2e05992ce0b4899de8a04f160c`.
- Frozen input receipt SHA-256: `350e7d4dec9f113d941f1415dc3bda4e556315ea57baffb89fe2efa62717a16b`.
- Feature-manifest SHA-256: `019cb10f6943c1e6b58ff671215e657d310a65cd53162538c4e3c3789d0694f6`.
- Feature cache: `C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-features-20261110-attempt-01\frozen-features.npz`, 78,306 rows × 384 FP32, SHA-256 `7b6cef6b6451eb7cafd8d918e34fe527fac20fb658ae79c692cac28c9130e4b1`.
- Frozen split: 72,010 train images, 1,000 selector images, 5,296 one-time reporting images; all have unique media identities. The 1,706 quarantined internal-validation items were not used. Official validation and sealed audit were not loaded.
- The readout used 288,040 image-task examples per epoch and 2,304,320 task presentations over eight epochs. There were 72,010 unique training images/assets and 288,040 unique image-task pairs; each pair was revisited across epochs.
- The reporting labels were loaded only after epoch 8 was selected and the best readout was reloaded. Reporting is a one-time internal validation measurement, not a blind or sealed audit.

## Runtime and exact recipe

- Encoder: OpenGVLab VideoMAE V2 ViT-Small K710, frozen, 22,153,286 parameters; source SHA-256 `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`; checkpoint SHA-256 `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- One static image was repeated across all 16 encoder frames. This contains no temporal evidence.
- Readout: 52,682 trainable parameters, seed 17, 8 epochs, batch 512 task examples, AdamW LR `1e-3`, weight decay `1e-4`, constant schedule, no early stopping. Selection was minimum equal-taxonomy selector NLL, earliest epoch on ties.
- Total exercised path: 22,205,968 parameters before quantization. Readout artifact 212,985 bytes; encoder weights 44,334,609 bytes; combined runtime weights 44,547,594 bytes. No quantization was run.
- Local hardware: NVIDIA GeForce RTX 3080 Laptop GPU. Feature extraction took 3,692.29 s (61.54 min), peak CUDA allocated 1,234,878,464 bytes. Readout training took 15.37 s; fit plus evaluation took 17.04 s, peak CUDA allocated 73,258,496 bytes. Local cost: no cloud or paid compute (`$0`).
- Software recorded by the run: Python 3.11.9, PyTorch 2.6.0+cu124. Deterministic training requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup.
- Feature extraction command:

  ```powershell
  python -m scripts.extract_videomae_clevr4_frozen_features `
    --archive C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\clevr_4_100k_v1.full.zip `
    --manifest C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03\features-manifest.jsonl `
    --output C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-features-20261110-attempt-01 `
    --batch-size 8
  ```
- Fit command (with `PYTHONPATH=src;.` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python):

  ```powershell
  python -m scripts.train_videomae_clevr4_option_scorer `
    --inputs C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03 `
    --features C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-features-20261110-attempt-01\frozen-features.npz `
    --output C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-decision-fit-20261110-attempt-02
  ```

## Learning curve

Training Accuracy is a rolling metric measured as weights change during the
epoch; it is not a fixed-checkpoint generalization-gap estimate.

| Epoch | Train CE | Rolling train Accuracy | Selector macro Accuracy | Selector macro NLL | Brier | ECE (15 bins) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.370 | 52.20% | 68.20% | 0.8828 | 0.3979 | 0.0850 |
| 2 | 0.740 | 73.02% | 74.30% | 0.6812 | 0.3192 | 0.0547 |
| 3 | 0.620 | 76.64% | 76.20% | 0.6118 | 0.2917 | 0.0377 |
| 4 | 0.570 | 78.22% | 77.70% | 0.5734 | 0.2752 | 0.0368 |
| 5 | 0.530 | 79.28% | 78.23% | 0.5486 | 0.2664 | 0.0386 |
| 6 | 0.510 | 80.03% | 78.50% | 0.5316 | 0.2595 | 0.0347 |
| 7 | 0.490 | 80.54% | 78.88% | 0.5156 | 0.2527 | 0.0313 |
| 8 (selected) | 0.480 | 81.02% | 79.62% | 0.5082 | 0.2495 | 0.0277 |

Selector metrics improved through the final scheduled epoch; no overfitting
signal appeared in this eight-epoch curve. Because the best point is the last
epoch and no longer schedule was run, this does not establish a plateau or
rule out later overfitting.

## Per-task metrics

All values are Accuracy, NLL, Brier, ECE (15 bins); each row contains one
decision per image for the named task.

| Task | Selector n=1,000 | One-time reporting n=5,296 |
|---|---:|---:|
| Color | 99.30%, 0.0324, 0.0137, 0.0173 | 98.70%, 0.0461, 0.0217, 0.0126 |
| Texture | 77.30%, 0.6351, 0.3084, 0.0434 | 79.04%, 0.5980, 0.2842, 0.0435 |
| Count | 42.60%, 1.3405, 0.6648, 0.0354 | 43.43%, 1.3152, 0.6585, 0.0210 |
| Shape | 99.30%, 0.0249, 0.0109, 0.0145 | 98.89%, 0.0342, 0.0171, 0.0081 |
| Equal-task macro | 79.62%, 0.5082, 0.2495, 0.0277 | 80.01%, 0.4984, 0.2454, 0.0213 |

Count is the clear bottleneck, with texture also substantially below the
per-task 95–98% aspiration. The large color/shape scores do not compensate for
those weaknesses. This result is compatible with a frozen representation or
readout limitation for count/texture, but the experiment does not distinguish
those causes from dataset/task properties. No architecture or data changes
were tested after seeing these results.

### Selector count error profile

This descriptive breakdown uses only the 1,000-image selector predictions;
the one-time reporting pool was not re-opened. Class supports range from 39 to
123, so the weakest recalls are not confined to the rarest classes.

| Target count | Support | Recall | Most common predictions when wrong |
|---:|---:|---:|---|
| 1 | 39 | 97.44% | 2 (1) |
| 2 | 90 | 73.33% | 3 (14), 1 (10) |
| 3 | 107 | 66.36% | 4 (18), 2 (16) |
| 4 | 101 | 41.58% | 3 (27), 5 (14), 6 (11) |
| 5 | 106 | 24.53% | 6 (37), 4 (24), 3 (10) |
| 6 | 123 | 46.34% | 5 (19), 8 (17), 7 (13) |
| 7 | 123 | 8.94% | 6 (44), 8 (34), 5 (15) |
| 8 | 95 | 27.37% | 7 (20), 6 (17), 10 (16) |
| 9 | 94 | 5.32% | 10 (38), 8 (32), 6 (12) |
| 10 | 122 | 68.85% | 8 (23), 6 (6), 9 (5) |

The errors often move to nearby counts, although 6/8/10 confusions are also
common. This points to a plausible loss of object-level spatial detail in a
single 384-wide global feature followed by the small MLP, but it does not prove
that mechanism. A future count-focused comparison would need a separate frozen
protocol that retains spatial/token features or supplies object-centric
features; changing the current run after observing the selector is not part of
this result.

## Artifacts, integrity, and deviations

The external run directory is
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-decision-fit-20261110-attempt-02`.
It contains the best and final readouts separately, all epoch history,
selector predictions, one-time reporting predictions, and `run-report.json`.
Key SHA-256 values:

- `best-readout.pt`: `437e920e7cda02ff2e28f19c96433af04aab3d70494030b6e3278809aa1cc39e`
- `final-readout.pt`: `1d6ff294efcf77cdd7448f6d64cf02382fb605ca21a761384fd3683733107046`
- `history.jsonl`: `32e20c615c123146272d11ba97b2bf8e9ecea7b3a726509091d02b5582b39bc4`
- `run-report.json`: `b28391777d26ce90b3c92be64df04c2fdbb6c4b005e78709d5b7c11f716628da`
- Selector predictions: `603e370f101d2c2f487757245e0a674b4ba1265d7cc995218c96d880d79a2d42`
- One-time reporting predictions: `4c39f57964c4440c7f2e71a5c35da9652459d14d99b8fc2634e603bda084461e`

An initial fit attempt (`attempt-01`) stopped on the first backward pass because
deterministic CuBLAS execution requires `CUBLAS_WORKSPACE_CONFIG` before
startup. It produced no training metrics or checkpoint. The same frozen inputs,
recipe, and seed were run successfully in separate `attempt-02` after setting
`:4096:8`; no optimizer state was resumed. No source code, input corpus,
legacy artifact, or sealed audit was changed or loaded.

## Next gate

Path A proves a pinned frozen-feature to JSON-option-scoring path runs at this
scale, but it does not pass a useful all-task quality gate. Preserve this result
as a baseline. Before broadening to more sources or adding a fusion module,
inspect count errors and feature separability on the existing frozen selector
predictions; any follow-up must use a newly frozen protocol and must not tune
against the one-time reporting pool. The overall four-modality product goal
remains unmet.
