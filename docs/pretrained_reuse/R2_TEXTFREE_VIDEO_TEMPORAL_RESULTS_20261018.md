# Text-free CLEVRER temporal-readout fit results (2026-10-18)

## Run outcome

The pre-registered matched fit completed on the local RTX 3080 Laptop GPU.
Both fresh-initialized readouts completed all 256 epochs / 2,048 optimizer
updates; selected and final weights are separate, and selected reloads reproduce
their validation logits exactly. The run report status is
`PASS_TEXTFREE_VIDEO_PAIRED_FIT`. No sealed audit was loaded and no product
checkpoint was changed or promoted.

This was a narrow, video-only CLEVRER probe. It does not test audio/video fusion,
open-ended video-language understanding, xLSTM, or the full Omni Decision model.
The input includes VideoPrism features, a typed symbolic program, taxonomy, and
supplied options, but excludes natural-language question text. The development
validation set selected checkpoints, so it is not a blind result.

## Frozen data and feature extraction

- Source: MIT-IBM/CLEVRER at revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0).
- Train manifest SHA-256:
  `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`.
- Validation manifest SHA-256:
  `53aa6f01447a43337a601279daf620c0df91d807773522cc92230fcab173db7c`.
- Train: 512 unique questions, 256 video scenes/assets. Validation: 400
  questions, 200 scenes/assets. Scene/media/content overlap is zero. Every
  selected media file was SHA-256 verified before fitting.
- VideoPrism-B model revision `be719a406d563b66f0ac969e7c94bab8e997c81a`,
  source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Eight 288px frames per asset; spatial tokens averaged per frame into `[8,768]`.
  CPU JAX 0.6.2 extraction took 654.75 seconds, peaked at 4,162,539,520 bytes
  working set, and produced a 10,457,419-byte feature cache.
- Feature cache SHA-256:
  `8878e2a51c9fb617466ec2e5b6dc174e9e7fb6eee666b6bee89b4071c2f00350`.
- There were 131,072 example exposures per arm (512 rows × 256 epochs),
  512 unique examples, 256 unique scenes/assets, and 130,560 repeat exposures
  after first use. Selected checkpoints consumed 13,312 examples / step 208
  (`mean_mlp`) and 9,216 / step 144 (`bilstm`); both had seen every unique
  training row by then.

## Fit settings and provenance

The arms differ only in temporal block: framewise MLP + mean pooling versus a
bidirectional LSTM. Each has the same query/program/candidate path. The MLP has
226,865 trainable parameters; the BiLSTM has 227,377 (+512, 0.23%). Both were
trained from fresh readout initialization with seed 17, the same epoch orders,
batch size 64, AdamW at 0.001, weight decay 0.0001, constant learning rate,
gradient clipping at 1.0, and no early stopping. The frozen VideoPrism encoder
has 114,365,184 parameters; isolated encoder-plus-readout counts are
114,592,049 and 114,592,561. These counts are not the full product model size.

- Source commit: `0020aa8f4428766d52ae1279b33574d7189009fc`.
- Config SHA-256: `39c947b4c65fe804c08de950c56f0443289483cea3b631c95747af5186844e33`.
- Effective config SHA-256:
  `7536a2246300ee2329482c64c2f4ce816afb6e491877a4505ec29ee9228f34b0`.
- Preflight SHA-256:
  `02c6f32101f553d050dea3ee5477bddedf5074dc417d2804fee47e841f5e119c`.
- Environment: Windows 10, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4,
  NumPy 2.4.6, NVIDIA GeForce RTX 3080 Laptop GPU.
- Per-arm training time: 19.64 seconds (`mean_mlp`) and 16.67 seconds
  (`bilstm`); complete run including validation, controls, and bootstrap took
  99.22 seconds. Peak PyTorch allocated CUDA memory was 150,212,096 and
  189,294,592 bytes, respectively. Cloud cost: 0.

Run artifacts are under
`C:/CodexArtifacts/pretrained-reuse-r2/textfree-video-clevrer-typed-fit-01/`.
The run-report SHA-256 is
`ebd35fb94cd655c982773e370200f987055a97a312536e9b3e0416a19d673365`.
The report contains complete learning histories, all per-example predictions,
selected/final checkpoint and prediction hashes, controls, and environment
metadata. No media or trained weight is committed to Git.

## Results

Metrics below use the validation-selected checkpoint. Accuracy, NLL, Brier,
and ECE are macro over the two question types; question-type rows show the
respective 200-example subset.

| Readout | Parameters | Best step / epoch | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE-15 | Temporal Accuracy / NLL | Static Accuracy / NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean MLP | 226,865 | 208 / 26 | 41.75% | 1.3080 | 0.6643 | 0.0965 | 36.50% / 1.3523 | 47.00% / 1.2637 |
| BiLSTM | 227,377 | 144 / 18 | 42.00% | 1.2661 | 0.6524 | 0.0830 | 38.00% / 1.2856 | 46.00% / 1.2466 |

Paired scene bootstrap (5,000 resamples, seed 17, percentile 95% interval)
for BiLSTM minus MLP:

| Question type | Accuracy difference (95% CI) | NLL difference (95% CI) | Brier difference (95% CI) |
|---|---:|---:|---:|
| Temporal descriptive | +1.5 pp (−3.5, +6.5) | −0.0667 (−0.1197, −0.0143) | −0.0333 (−0.0585, −0.0080) |
| Static descriptive | −1.0 pp (−6.5, +4.5) | −0.0171 (−0.0649, +0.0286) | +0.0095 (−0.0122, +0.0310) |

The temporal NLL and Brier intervals favor the BiLSTM on this selected
development validation; its temporal Accuracy interval includes zero. Static
differences are uncertain. Aggregate improvement is small and does not establish
a general recurrent-memory benefit.

## Overfitting and input-use controls

Both arms reached 100% training Accuracy and near-zero final train NLL. Their
validation-selected epochs were early: at selection, MLP train Accuracy/NLL was
59.38% / 0.9510 and BiLSTM was 53.32% / 1.0577. At the final checkpoint,
validation metrics were:

| Readout | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE-15 | Temporal Accuracy / NLL | Static Accuracy / NLL |
|---|---:|---:|---:|---:|---:|---:|
| Mean MLP | 38.75% | 4.3329 | 1.0768 | 0.4962 | 31.50% / 5.0435 | 46.00% / 3.6223 |
| BiLSTM | 39.75% | 4.1896 | 1.0568 | 0.5003 | 34.00% / 4.6755 | 45.50% / 3.7037 |

The predefined overfit warning triggered for both arms: train NLL continued to
fall while validation NLL rose substantially. Final weights were preserved and
not substituted for the selected checkpoints.

At the selected checkpoint, zeroing or scene-shuffling video substantially
reduced performance, showing the model used aligned video features. For the
BiLSTM, aligned versus scene-shuffled validation differences were +4.5 pp
temporal Accuracy (95% CI −3.0 to +12.0), −0.0690 temporal NLL (−0.1261 to
−0.0138), and −0.0457 temporal Brier (−0.0810 to −0.0102). The aligned and
reversed-frame BiLSTM scores were almost identical (42.00% / 1.2661 NLL versus
42.75% / 1.2656), so this probe does not show a useful frame-order advantage.
The MLP was permutation-invariant as expected.

## Decision and limits

Do not adopt the BiLSTM as a general temporal component from this probe alone.
It delivered a small temporal calibration/loss improvement on one seed and one
development validation split, while accuracy improvement was uncertain and
static descriptive results did not improve. Strong late overfitting limits the
use of the final checkpoint, and the early selected checkpoint used only a
small fraction of the 2,048-update budget. This does not prove a data or model
capacity ceiling and is not evidence of grokking.

The next decision should inspect per-type errors and task encoding, then test
whether the result replicates under a preregistered regularized fit or a
representative video task mix before changing the full multimodal architecture.
Do not tune against sealed audit, add xLSTM, or promote a product checkpoint on
these results. The 95–98% four-modality objective remains open.
