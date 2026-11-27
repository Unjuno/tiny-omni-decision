# R2 IntPhys Path A temporal-prediction diagnostic

**Status: fixed Path A diagnostic completed on real IntPhys train/dev data.**

## Purpose and scope

This is a task-specific Path A experiment for temporal physical-plausibility
judgment. It reuses the pinned frozen VideoPrism-B frame representation and
fits a small causal LSTM predictor on possible-only IntPhys training videos.
The scorer is the mean cosine error between predicted and observed next-frame
features. It is an anomaly score, not a general Decision model and not a
calibrated probability over JSON options.

The implementation does not add a BERT text model, shared multimodal Lens, or
sensor input. Those require separate admitted tasks and evidence. Existing
BERT-Mini results are task-specific and do not justify broadening this video
experiment.

## Data gates

- The source is the official IntPhys training shard (`train.1.tar.gz`),
  declared CC BY 4.0 by the publisher. The extraction refuses any size or
  published MD5 mismatch before reading the archive.
- Train status metadata must mark every record possible and describe exactly
  100 frames. Each record must have all 100 scene, depth, and mask indices.
- Eight uniformly spaced RGB frames feed the frozen feature extractor. All
  100 original RGB PNG payloads are SHA-256 hashed in temporal order for
  duplicate grouping and split integrity. Only the eight sampled RGB frames
  are decoded for feature extraction; the remaining PNG payloads are byte-
  hashed but not decoded, which remains a data-quality limitation.
- The deterministic internal holdout groups duplicate full status metadata,
  identical complete RGB byte sequences, or identical normalized eight-frame
  model inputs before splitting. Keeping both RGB identities catches exact
  duplicates even if PNG re-encoding changes bytes, while the sampled-input
  check avoids leakage between videos indistinguishable to the actual encoder
  input. It is used only for training early stopping and threshold fixation.
- IntPhys dev is a fixed development diagnostic, not a blind sealed audit. It
  contains 90 scene groups, each with four renditions and a 2 possible / 2
  impossible label balance. The extractor reports exact cross-label collisions
  among the eight sampled RGB inputs. Labels do not enter fitting, checkpoint
  selection, or score-threshold selection.
- The official hidden test is not accessed.

## Model and evaluation protocol

The train-only runner selects the best checkpoint by internal validation
cosine prediction loss. The evaluator validates pinned source, encoder, archive,
feature, and checkpoint receipts before scoring. It rejects sample-ID,
scene-group, and normalized sampled-input hash overlap across train, internal
validation, and dev. It compares the learned LSTM against a
last-feature persistence baseline. Each model's binary threshold is fixed
independently at the 95th percentile of its internal possible-only validation
anomaly scores.

The report includes ROC AUC, accuracy, sensitivity, specificity, and balanced
accuracy overall and by IntPhys block. A paired bootstrap resamples complete
scene groups to estimate uncertainty for learned-minus-persistence differences.
Per-example IDs, labels, and raw anomaly scores are saved. NLL, Brier, and ECE
are intentionally omitted because neither anomaly score is a calibrated class
probability.

## Real-data result (2026-11-27)

The official `train.1.tar.gz` download completed at 37,016,506,106 bytes.
MD5 is `3428bcf398441a489111528634530212`; SHA-256 is
`5f4ce9b2240bdbd57e32ed2d8315300314d30572f39adb0d9e58afa3b27da83a`.
The source declares CC BY 4.0. The pinned VideoPrism checkpoint SHA-256 is
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
source revision is `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.

| Split | Videos | Unique scene/media groups | Feature file SHA-256 |
| --- | ---: | ---: | --- |
| Train | 3,004 | 3,004 | `fa2655923d4ced7ab479a2eb47ac4cd8eb3e10238a860defd21b2effcc9c11b6` |
| Internal validation | 746 | 746 | `ff7c9c27e91d5fa8ebf6437e6e0aa35f3d79830c38b5b6d9502dab787b331adc` |

The source contains 3,750 videos and 3,750 unique status/media groups. All
records are possible-only. Scene, depth, and mask channels each have 375,000
frames; each video was checked for the complete 1–100 index range. Duplicate
groups and train/validation overlap are zero for status identity, complete
ordered RGB identity, normalized eight-frame model input, sample ID, and group
ID. Features have shape `[video, 8, 768]`; only eight evenly spaced RGB frames
were decoded, while all 100 RGB PNG payloads were hashed for identity.

The recovered train extraction report is at
`C:/CodexArtifacts/pretrained-reuse-r2-local/intphys-patha-training-20261127-attempt-01/train-features-seed17/extract-report.json`.
The run wrote both feature NPZs after all data and split gates passed, then
failed while recording TensorFlow's distribution version: this Windows build
registers `tensorflow_cpu`, while the imported module reports
`tf.__version__ == 2.18.1`. The files were preserved, re-hashed, and independently
revalidated before a receipt was recovered. Exact extraction duration, model
load time, per-video latency, and peak memory counters were not retained and
remain UNKNOWN; the observed process start was 08:02:30 JST and feature files
were written at 09:36:40–41 JST. The extractor now records the imported module
version with a distribution-metadata fallback.

### Training

One seed-17 CPU run used 3,004 train / 746 validation sequences, hidden width
64, AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch size 64, maximum 100 epochs,
patience 8, and `min_delta=1e-4`. It completed 83 epochs and 3,901 updates in
69.21 s on Python 3.11.9 / PyTorch 2.6.0+cu124. The trained predictor has
264,960 parameters. The selected best checkpoint is epoch 81 by raw minimum
validation cosine loss `0.0025656382`; the 95% possible-only validation-tail
threshold is fixed only in the later evaluator.

Train loss fell from `0.38902` at epoch 1 to `0.00246186` at epoch 83. Validation
loss fell from `0.0328422` at epoch 1 to `0.00256564` at selected epoch 81, then
was `0.00257834` at epoch 83. This curve does not show strong overfitting within
the measured budget, though the train and validation data are from the same
source distribution and this is not a generalization guarantee.

The first saved run selected epoch 75 because `min_delta` was incorrectly used
for both patience and best-checkpoint replacement, even though epoch 81 had a
lower raw validation loss. That artifact remains preserved and was not used for
evaluation. The selector was corrected: every raw validation minimum updates
the best checkpoint, while `min_delta` only controls patience. A fresh output
directory was used; the corrected run reproduced the same training history and
selected epoch 81. Best checkpoint SHA-256 is
`11966b85f5148a166e309ae47f4566617a7ebbd1ad661df24287a18df37a7d2d`; history
SHA-256 is `c24206d4e1c572e0931a3c263b44652b3cddfde0ac449e496ab889c5a835d2ea`.

### Fixed dev diagnostic

The pinned dev partition has 360 videos in 90 scene groups, with four
renditions and a 2 possible / 2 impossible balance per group. Its archive
SHA-256 is `441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a`.
The tar members are interleaved across videos rather than grouped. The first
extractor attempt exposed this format mismatch; the extractor now groups
members by `(block, scene, rendition)`, validates all three 100-frame channels,
and keeps selected PNG bytes until features are computed. The fixed dev
features were extracted once with the pinned CPU encoder in 464.4 s; per-video
forward median/p95 were 1.164 / 1.230 s and peak working set was 4.49 GB.

The sampled-input collision audit found 9/90 scene groups where a possible and
impossible rendition have exactly identical normalized eight-frame inputs
(18 opposite-label sampled-input signatures). These are genuine
representation/sampling ambiguities. They were retained; no labels/examples
were removed.

After the candidate and threshold rule were frozen, one 2,000-replicate,
seed-17 paired scene-group bootstrap evaluation compared the LSTM and
last-feature persistence. Overall:

| Model | Accuracy | Balanced accuracy | ROC AUC | Sensitivity (impossible) | Specificity (possible) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.5000 | 0.5000 | 0.4919 | 0.4667 | 0.5333 |
| Persistence | 0.4889 | 0.4889 | 0.4921 | 0.1500 | 0.8278 |

The paired difference in balanced accuracy was `+0.0114` with 95% interval
`[-0.0056, +0.0306]`; ROC AUC difference was `-0.00075` with interval
`[-0.0145, +0.0137]`. Both intervals include zero; this does not establish an
improvement over persistence, and both models are near chance. Per-block AUC
was LSTM `0.4775 / 0.4906 / 0.5081` versus persistence `0.4417 / 0.4972 /
0.5233` for O1/O2/O3. NLL/Brier/ECE are omitted because these residual scores
are not calibrated class probabilities.

This is one known-development evaluation, not a blind audit, and it does not
show that the 100–200M Omni Decision objective is achieved. The result is **not
supported** for the hypothesis that this LSTM improves physical-plausibility
judgment over persistence under the current eight-frame representation. It is
not evidence of a general architecture ceiling. The data point toward temporal
aliasing and a feature/task mismatch as follow-up questions; more BERT work is
not a direct remedy for this video result. BERT remains a separate text-task
candidate and must be compared on defined text Decision tasks.

## Artifacts and integrity

All outputs are under `C:/CodexArtifacts/pretrained-reuse-r2-local/` in new
attempt directories; media, feature arrays, checkpoints, and per-example
predictions remain outside Git. Key locations:

- Train features and recovered extraction receipt:
  `intphys-patha-training-20261127-attempt-01/train-features-seed17/`
- Corrected selected model, metadata, and history:
  `intphys-patha-training-20261127-attempt-01/lstm-predictor-seed17/`
- Fixed dev archive, extracted features, and audit:
  `intphys-patha-data-quality-20261127-attempt-01/` (`dev.tar.gz`,
  `dev-features-seed17/dev-features.npz`, and `dev-audit.json`)
- One-shot evaluation and row-level predictions:
  `intphys-patha-data-quality-20261127-attempt-01/evaluation-seed17-best81/`

The selected checkpoint SHA-256 is
`11966b85f5148a166e309ae47f4566617a7ebbd1ad661df24287a18df37a7d2d`; training
history SHA-256 is
`c24206d4e1c572e0931a3c263b44652b3cddfde0ac449e496ab889c5a835d2ea`; training
metadata SHA-256 is
`9a3ef91ce0b946a39a4179d37ca763d69fae206076e3407959a0f9af291efae9`. The
evaluation report SHA-256 is
`e0a9bb735ebc5acc1d0597301908f1989fddb0258e53b6a4d2819dcb004e2a34`, and the
predictions SHA-256 is
`65910363eaf558fa8a89e458db65814edbb8570f7eb9fafb388bfb5f313e72be`. Raw
archive and feature hashes are recorded above. The hidden/sealed test was not
accessed. No cloud or GPU compute was used; no live process, old worktree,
legacy model, or checkpoint was changed.

The work remains on `codex/r2-intphys-patha-20261127`. Focused tests cover
Windows runtime-version metadata, interleaved dev members, all-channel frame
completeness, and raw-best checkpoint selection. Local verification: `uv run
pytest -q` passed 3,004 tests (11 skipped); `uv run ruff check .` passed; model,
dataset, candidate, source-catalog, and durable-teacher manifest validations
passed. `git diff --check` passed. Push and CI status will be recorded after
branch verification.

## Next evidence-driven step

Do not widen this video path into BERT, model merging, or a shared Lens based on
these near-chance results. First run a controlled temporal-resolution/data
diagnostic that changes only the number of sampled frames, keeps source records,
model revision, split, seed, and readout fixed, and measures whether the 9 exact
input-collision groups disappear and dev discrimination moves above persistence.
If that fails, stop this feature path and screen a different admitted physical
reasoning representation/task before increasing capacity. Do not inspect the
sealed test or promote this predictor as a product Teacher.
