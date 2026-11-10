# VideoMAE spatial count readout — 2026-11-10

## Decision

Do not adopt the tested 4×4 spatial-grid linear readout. On a fixed, balanced,
train-only 2,000-image holdout it gained 1.60 percentage points of count
Accuracy over the matched token-mean control, but the paired 95% interval spans
zero (−0.70 to +3.90 points). Its NLL and Brier were substantially worse, and
its train/holdout gap was large. This single diagnostic does not establish a
VideoMAE representation limit, nor does it justify more tuning on these IDs.

This is a narrow Clevr-4 synthetic object-count result. It is not temporal video
reasoning, general image understanding, a four-modality result, or product
quality evidence. No candidate is promoted.

## Frozen protocol and isolation

- Source branch/commit: `codex/r2-videomae-count-feature-separability-20261110`
  at start commit `5abf2429a0b987ff28092f693cc14142f1359732`.
- The extractor/readout implementation is committed in this report's branch;
  its final source SHA-256 is
  `e3a14a699cf115875de2d048fb00befc27894588821b6d2daab72c109d7c4c4f`, and
  focused test source SHA-256 is
  `7b86348bfd335dcb35da4b5cab384d78378b804ca6ef93cdb44d326f065cd863`.
- Encoder: OpenGVLab VideoMAE V2 ViT-Small K710, frozen; 22,153,286
  parameters; source SHA-256
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`;
  checkpoint SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Source archive: 36,057,238,315 bytes. The pinned source receipt has SHA-512
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`;
  its full SHA-512 was not recomputed. Each of the 10,000 consumed image
  payloads passed its individual SHA-256 check.
- Frozen train input hashes: media manifest
  `e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410`, labels
  `45de9d36d557619789dd6160e808f7ff0c531f06a298f0b2af80c9720103b695`.
- Selection: within each count class 1–10, rank official-train image IDs by
  SHA-256 of `17:select:<id>`, take 1,000, assign the first 800 to fit and last
  200 to holdout. Thus 8,000 fit and 2,000 holdout images, exactly 200 per
  class in holdout. Image/asset IDs and media hashes are unique and split
  overlap is zero. The frozen ordered manifest hashes to
  `47ecadd040371384a1cfa57ef9676cd2459a9266553d6bc99686eb486b2a097b`.
- This split was constructed from official TRAIN records only. It is new to
  this readout fit, but these records and the count task were used in an earlier
  full-corpus experiment. The split is therefore development evidence, not a
  history-blind evaluation. The first pilot (attempt-01) exposed these same
  holdout labels before attempt-02 corrected a normalization confound. The
  holdout is now considered exposed and must not be reused for further tuning.
- The 1,000-image selector, 5,296-image one-time reporting pool, official
  validation, and sealed audit were not read. Existing feature cache,
  checkpoints, model, and other runs remained read-only.

## Compared representations and fit

Every condition used the same seed (17), same 8,000/2,000 IDs, train-only
feature standardization, one linear 10-class head, AdamW (`lr=0.01`,
`weight_decay=0.001`), and exactly 100 full-batch epochs. The final epoch was
used; no validation checkpoint selection or hyperparameter search occurred.

| Condition | Feature construction | Head parameters | Total pre-quantization parameters including encoder |
|---|---|---:|---:|
| `global_384` | Standard VideoMAE mean-pool then `fc_norm` | 3,850 | 22,157,136 |
| `token_mean_384` | Apply `fc_norm` to each final token, then average 8 temporal tubelets × 14 × 14 positions | 3,850 | 22,157,136 |
| `spatial_grid_4x4` | Same per-token `fc_norm`, average tubelets, adaptive-average-pool the 14×14 map to 4×4, flatten | 61,450 | 22,214,736 |

The token-mean control isolates normalization order from spatial preservation.
The spatial head has 16× the readout parameters of either 384-D head, so spatial
layout and head capacity are not independently isolated. No claim of parameter
efficiency follows from this comparison.

Attempt-01 used the same fixed data and recipe but compared standard global
pooling directly against per-token-normalized spatial features. That was a
real exploratory pilot, but the comparison changed normalization order as well
as feature layout. Its artifacts remain preserved; attempt-02 is the corrected
primary diagnostic. The shared holdout exposure is disclosed above.

## Results

| Representation | Train Accuracy | Holdout Accuracy | Holdout NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Standard global 384 | 54.15% | 38.75% | 1.4852 | 0.7206 | 0.0648 |
| Token-mean 384 | 55.09% | 38.70% | 1.4714 | 0.7168 | 0.0687 |
| Spatial grid 4×4 | 96.69% | 40.30% | 1.9614 | 0.8419 | 0.2949 |

Per-count holdout Accuracy (200 examples per class):

| Count | Global 384 | Token-mean 384 | Spatial 4×4 |
|---:|---:|---:|---:|
| 1 | 85.0% | 87.5% | 89.0% |
| 2 | 52.5% | 49.5% | 54.5% |
| 3 | 44.0% | 43.0% | 42.0% |
| 4 | 38.5% | 35.0% | 35.5% |
| 5 | 22.0% | 24.0% | 27.5% |
| 6 | 21.5% | 23.5% | 26.0% |
| 7 | 26.5% | 25.0% | 28.5% |
| 8 | 24.0% | 23.5% | 19.5% |
| 9 | 19.5% | 20.5% | 24.5% |
| 10 | 54.0% | 55.5% | 56.0% |

Paired bootstrap used 10,000 resamples of unique image/asset rows, seed 17,
percentile 95% intervals:

| Difference | Estimate | 95% interval |
|---|---:|---:|
| Spatial − token-mean Accuracy | +1.60 pp | −0.70 to +3.90 pp |
| Spatial − token-mean NLL | +0.4900 | +0.4190 to +0.5630 |
| Spatial − token-mean Brier | +0.1251 | +0.1022 to +0.1482 |
| Token-mean − standard-global Accuracy | −0.05 pp | −1.25 to +1.10 pp |
| Token-mean − standard-global NLL | −0.0138 | −0.0226 to −0.0050 |

Spatial training CE fell from 7.755 (epoch 1) to 0.294 (epoch 100), while
holdout Accuracy stayed at 40.30%; the corresponding train Accuracy was
96.69%. This is strong overfitting under the fixed recipe. Accuracy alone
does not establish a benefit: the paired interval includes zero, while
probability quality is clearly worse. No epoch extension or regularization
search is authorized on this exposed holdout.

## Execution and artifacts

- Preparation command:

  ```powershell
  python -m scripts.videomae_spatial_count_probe prepare `
    --inputs C:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03 `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-01\prepared
  ```

- Extract command (attempt-02, 10,000 images):

  ```powershell
  python -m scripts.videomae_spatial_count_probe extract `
    --prepared D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-01\prepared `
    --archive C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\clevr_4_100k_v1.full.zip `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-02\features `
    --batch-size 8
  ```

- Fit and prediction command:

  ```powershell
  python -m scripts.videomae_spatial_count_probe analyze `
    --prepared D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-01\prepared `
    --features D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-02\features `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-02\analysis
  python -m scripts.videomae_spatial_count_probe summarize `
    --predictions D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-02\analysis\holdout-predictions.jsonl `
    --output D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-02\statistics-final
  ```

- Local hardware/software: RTX 3080 Laptop GPU 16 GiB, Python 3.11.9,
  PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6, Pillow 10.4.0. No cloud or
  paid compute. Extraction times were 487.918 s (attempt-01) and 479.879 s
  (attempt-02); the 32-image smoke took 2.201 s of measured extraction time.
  Model-load time was not included in those extraction timers. Peak CUDA
  allocation was 1,260,122,112 bytes. Fit plus the first summary command took
  about 10.2 s wall-clock; per-condition fit times were not separately logged.
- Attempt-02 feature archive: 256,659,120 bytes, SHA-256
  `6524bf05df378866a83f09a5bc9b35277f16cce162411ec523e7a31d00583ed4`.
  Holdout predictions SHA-256
  `3d00b3347d7c7334cb32ac3412ea30d04a0379833bdfd0be44bfe9d1ee3da1df`.
  Final paired statistics SHA-256
  `075aeee7b6cee84afdff5327cfa8cd759f0777cbd1d9c7eac71df557e1b939fb`.
- Checkpoints are external under
  `D:\CodexArtifacts\pretrained-reuse-r2\videomae-spatial-count-probe-20261110-attempt-02\analysis\`:
  `global_384.pt` (20,520 bytes), `token_mean_384.pt` (20,552 bytes), and
  `spatial_grid_4x4.pt` (297,048 bytes). They contain only linear readout
  weights and train-only standardization values; the encoder remains the
  existing read-only checkpoint. Full feature archive and predictions remain
  outside Git. Their SHA-256 values are `40df71ff894a9d59cac05103477c1879bf688d44e38e46dc04ff718bee29d6cd`,
  `c5a26302e98c9c9fde0c12bd7d6251fd73246bab8eb41fc90163eded24d3ee88`, and
  `5009e1d71b5421691eae7c7e680d996c511752b4d00b6eb17796cc2044b89d26`,
  respectively.
- At the time of final resource sampling, C: had 0.94 GiB free, D: had
  approximately 2.05 TiB free, and the GPU was idle. Results were written to D:.

The first 32-image smoke succeeded with the pinned source and checkpoint. Six
focused tests pass. Full-suite verification first had three unrelated Windows
fixture path failures because the chosen temporary path was too deep; those
exact tests passed under a short D: path. The final full-suite retry passed:
`2,895 passed, 11 skipped` using a short D: pytest temp directory. `ruff check .`
and `git diff --check` passed. Commit `86a849443cdc5bf17cdc007d43d8c761cd0611cf`
was pushed to the dedicated branch; GitHub Actions run
[`42772649950`](https://github.com/Unjuno/tiny-omni-decision/actions/runs/42772649950)
completed successfully. No model weights, media, or large feature cache were
added to Git.

## Interpretation and next action

The spatial grid gave no reliable Accuracy improvement over the token-mean
control and materially worsened NLL, Brier, calibration, and the train/holdout
gap. Stop spatial-grid tuning on these exposed IDs. This does not show the
pretrained representation lacks object-count information: a different frozen
readout, regularization, or object-centric feature could behave differently,
but evaluating those on this now-exposed holdout would be selection leakage.
Any later count experiment needs a newly frozen, historically audited pool or a
different admitted task. The overall Text/Image/Audio/Video Omni Decision goal
and the R2 modality-native gates remain open.
