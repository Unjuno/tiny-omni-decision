# AVATAR native spatial localization probe (2026-11-02)

## Scope and status

This is a frozen-feature, TRAIN-only development experiment. It converts a
single-frame AVATAR sound-source annotation into a 40-option coarse spatial
Decision: given the synchronized frame and its centered two-second audio
embedding, predict the 8 by 5 image cell containing the annotated bounding-box
center. Each validation prediction is saved as JSON with the 40 cell
probabilities. `Single-Sound` frames with multiple boxes retain all target
centers; two examples have multiple boxes. Category labels are not model inputs.

The 20% development-validation partition was deterministically grouped by the
underlying source-video identity from the original TRAIN annotation folder.
Official validation and sealed audit were not opened. This validation is not a
blind test and does not establish product performance. No ternary conversion,
encoder fine-tuning, or product-model promotion occurred.

## Data and split

The builder read 19,159 annotation JSON files under the already available
AVATAR `train/` directory and retained 12,004 usable `Single-Sound` records.
The resulting fold has 9,693 fit examples from 1,817 source groups and 2,311
development-validation examples from 454 source groups. Those rows cover 2,132
and 510 clips respectively. Source-group overlap and frame-asset overlap are
both zero. The annotation digest is
`8f5f7c4ff758ee14db1afbbeecaea7886ba5f5d580b776bc2763ef930eaf1b95`; exact
split hashes and counts are in the machine-local fold manifest.

The existing TRAIN-only EfficientAT cache has 11,955 matching frame windows.
All 11,955 audio centers exactly match the frame annotation time. We retained
all examples: 38 fit and 11 validation rows without cached audio receive a zero
audio vector plus an explicit missingness bit. Those 49 rows are reported and
included in the main comparison; available-only summaries are also retained.

## Frozen encoders and extraction

| Component | Revision / weights | Parameters | Role |
|---|---|---:|---|
| DINOv2-base | `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`; SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841` | 86,580,480 | Per-frame 16 by 10 patch tokens pooled to 8 by 5 |
| EfficientAT DyMN-0.4 | `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806`; SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | 1,961,239 | Existing centered two-second audio features |

DINO input is letterboxed to 224 by 140, preserving aspect ratio; positional
embeddings are interpolated. No media was written by the extractor. It produced
12,004 finite spatial feature maps of shape `5x8x768`; maximum frame-time error
was `1.78e-15` seconds. Feature-cache SHA-256 is
`df25d3a18f5873e2927cd5b733102fc017bc9e62170802a29db087b7b540b278` and size is
682,195,100 bytes. A separate two-clip/10-frame CUDA smoke also passed.

The full extraction ran through the native HotGPU reservation on the local RTX
3080 Laptop. The GPU was idle before reservation; Orbitune was not stopped or
modified. CUDA 12.4, PyTorch 2.6.0+cu124, Transformers 5.6.2, and PyAV 18.1.0
were used. Peak PyTorch CUDA allocation was 442,728,448 bytes. Extractor time,
including final cache and archive-hash work, was 2,909.389 seconds; no cloud
compute or cost was used.

Before the full job, three TRAIN-derived media samples measured total decode,
preprocessing, and forward times of 0.2666, 0.3051, and 0.1065 seconds (mean
0.2261 seconds). The first three full-run progress intervals measured 105.1,
108.3, and 106.7 seconds per 100 clips (mean 106.7). The later intervals stayed
near that rate. All observed checkpoints and the three sample details are in
the external timing receipts.

## Readout protocol

Frozen features were fitted on CPU with seed 17. The 40 candidate cells are
row-major. Models were selected only by development-validation target-set NLL
(`min_delta=0.001`, patience 10, maximum 100 epochs). The baselines are a
smoothed fit-set spatial prior, a Video-only shared cell scorer, an Audio-only
40-cell classifier, and a low-rank Audio-Video interaction scorer. The joint
readout has 37,704 trainable parameters; no encoder or feature projector was
updated. The Video-only and Audio-only heads have 808 and 15,440 parameters.

NLL scores the probability mass assigned to the annotated center cell(s).
Brier uses a uniform target distribution over those cells. ECE has 10 bins and
measures top-cell confidence against whether that cell matches the target set.
Predictions, per-source-group metrics, epoch histories, best and last weights,
and a validation group-paired bootstrap are saved outside Git.

## Results

| Readout | Fit-selected epoch | Accuracy / hit@1 | Hit@5 | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smoothed fit-set spatial prior | — | 22.07% | 74.82% | 2.342 | 0.852 | 0.051 | 0.272 |
| Video-only | 6 / 16 | 45.39% | 90.78% | 1.751 | 0.726 | 0.140 | 0.594 |
| Audio-only | 1 / 11 | 21.72% | 65.30% | 3.059 | 0.983 | 0.243 | 0.460 |
| Joint Audio-Video | 1 / 11 | 39.94% | 82.82% | 5.603 | 1.044 | 0.481 | 0.880 |

The joint-minus-Video-only hit@1 difference is `-5.45` percentage points. A
paired source-group bootstrap (2,000 resamples, seed 17, 454 validation groups)
gives a percentile 95% interval of `[-9.09, -1.72]` points. The selected joint
checkpoint is severely overconfident and has a 2.48 NLL train-to-validation
gap; the corresponding gap is 0.42 for Video-only and 0.83 for Audio-only.

The joint checkpoint reacts to audio: shuffling audio embeddings among the
available validation examples, while preserving missingness, lowers hit@1 to
27.52% and raises NLL to 7.615. But zeroing audio content while keeping the
missingness bit raises hit@1 to 42.80% and lowers NLL to 3.316, still worse
than Video-only. Audio-only validation hit@1 is 21.72%, close to the 22.07%
spatial prior. These controls show audio dependence, but no useful generalized
audio contribution in this readout. The 11 audio-missing validation examples
are too few for a separate quality claim.

Source-group macro hit@1 is 43.69% Video-only, 21.34% Audio-only, and 38.04%
Joint. Source-group macro NLL is 1.813, 3.127, and 5.898 respectively. Full
per-group values and audio-availability strata are in
`source-group-and-availability-metrics.json` outside Git.

## Interpretation and next work

This result does not support promoting the current Audio-Video readout. The
Video-only result demonstrates a useful spatial signal beyond the train-set
location prior, while the EfficientAT embedding alone contributes little to
localization. The joint interaction is audio-sensitive but generalizes worse,
with pronounced calibration and train/validation gaps. This is evidence against
this specific readout and feature pairing, not a general limit on multimodal
fusion or DINOv2.

The frozen encoders total 88,541,719 parameters; with the joint head the
pre-quantization count is 88,579,423, below the provisional 200M parameter
budget. The DINOv2 checkpoint itself is larger than the provisional 200 MB
package target; no quantized or deployable package was produced, so that size
gate remains unmet. This single AVATAR task does not address text, speech,
general video reasoning, or the broader 95–98% objective.

The next justified work is to inspect audio-feature coverage/semantics and
compare one more validation-only, better-regularized interaction readout with
the same frozen data and split. Keep the Video-only model as the baseline; do
not tune against or open sealed audit.

## Machine-local artifacts

All bulky files remain under
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-native-localization-20261102-attempt-01\`:

- `fold/` — deterministic TRAIN-only split and manifest
- `spatial-features.npz` and `spatial-features.metadata.json` — DINO spatial cache
- `spatial-cache-validation.json` — ID, split, shape, and finiteness checks
- `three-sample-throughput.json` and `spatial-extraction-progress.json` — measured timings
- `readout-01/` — effective config, training history, best/last heads, predictions, result and source-group metrics

No annotation media, feature cache, model weights, predictions, or large
checkpoint was added to Git.
