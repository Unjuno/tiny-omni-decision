# AVMIT cross-clip audiovisual Decision protocol — 2026-11-10

## Status and scope

This protocol is fixed before any AVMIT feature record is decoded and before
model fitting. It is a development-only audiovisual cross-modal diagnostic,
not a product-model qualification. AVMIT supplies precomputed YAMNet and
EfficientNet-B0 features; these are not the selected Path A feature producers.
No original Moments in Time video/audio media, published test examples,
sealed-audit examples, or sealed labels may be read. This experiment cannot
close R2 or support a four-modality quality claim.

The Zenodo record reports CC BY 4.0 for the released annotation/feature files.
The original media have separate rights and are not part of this experiment.
Retain attribution to AVMIT and the release authors in any report using its
features.

## Task and data unit

The binary supplied-option Decision asks whether two **different clips** have
the same AVMIT action label:

- `same_action`: audio and video sources are different clips whose AVMIT labels
  agree;
- `different_action`: audio and video sources are different clips whose
  labels differ.

Every pair group contains four distinct clips: two clips from each of two
labels. Its eight rows include cross-clip positives and negatives. Within a
group, every source appears once under each target as audio and once under
each target as video. A clip is used in at most one group per split. This
prevents a single audio or video feature from identifying the target and
prevents same-record audio/video synchronization from defining positives.
The independent uncertainty unit is the eight-row pair group.

The only included manifests are `pairs-v3/{train,calibration,validation}.jsonl`
under the external attempt directory. The earlier same-clip-positive `pairs/`
version is retired and must not be read for fitting or evaluation. The sealed
split is not an input to any command in this protocol.

| Split | Independent groups | Rows | Unique source clips |
|---|---:|---:|---:|
| Train | 9,552 | 76,416 | 38,208 |
| Calibration | 1,317 | 10,536 | 5,268 |
| Development validation | 1,374 | 10,992 | 5,496 |

## Feature and model comparison

After verifying the full archive size and publisher MD5, read only the unique
feature records referenced by the three allowed pair manifests. Verify each
TFRecord framing checksum, the embedded source filename, nonempty finite
float32 sequence features, constant width within each stream, and exact
one-record-per-clip correspondence. Keep time-step order; do not inspect or
export other archive members' payloads. Cache only feature arrays and hashed
source IDs outside Git.

Compare exactly two predeclared readouts, both using the same frozen features,
pair groups, optimizer, seed, and option scorer:

1. `mean`: independent per-stream linear projection to 64 dimensions, then
   masked temporal mean pooling;
2. `gru`: independent one-layer per-stream GRUs with hidden width 64, returning
   the final valid state.

Both variants form the same cross-modal interaction vector
`[audio, video, abs(audio-video), audio*video]` and score the two supplied
typed options with one shared 128-hidden-unit candidate scorer. No raw text
encoder, feature fine-tuning, or extra task labels are used. Record exact input
dimensions, trainable parameter counts, and serialized readout sizes after the
feature-shape gate.

Decision contract (fixed order for storage; test reversed order at inference):

- question: `Do these two different clips have the same AVMIT action label?`
- options: `same_action` / `different_action`
- target: pair-manifest target

## Training and selection

- Seed: 17 for Python, NumPy, and PyTorch.
- Train only on the 9,552 training groups; shuffle groups, never individual
  rows. Keep all eight rows of a group in the same minibatch.
- Batch: 64 groups (512 rows); AdamW with learning rate `1e-3`, weight decay
  `0.01`, gradient norm cap `1.0`; cross-entropy only; at most 20 epochs.
- After each epoch score the independent calibration groups. Preserve the best
  checkpoint by lowest calibration NLL; stop after four consecutive epochs
  without improvement. Keep the final epoch separately.
- Fit one scalar temperature on calibration predictions for the selected
  checkpoint. Report both uncalibrated and temperature-scaled results.
- Validation is evaluated once after both configurations and calibrators are
  frozen. Do not change features, architecture, optimizer, or selector based
  on these validation scores.

Before full fitting, pass an 8-group deterministic forward/backward/save/reload
smoke with finite loss, unchanged frozen feature tensors, option-order parity,
and measured VRAM. A smoke failure blocks full fitting; it does not authorize
changing the protocol silently.

## Controls and metrics

The paired construction gives a structural unimodal ceiling: within each
group, each exact source audio and video feature is repeated once under each
target. A model receiving only one source feature cannot distinguish that
source's two target rows; report the resulting 50% row-accuracy control and
chance probability baseline, with the proof checked from manifests. Also
verify all four within-group cross-clip combinations, option-order
equivariance, and that the runtime evidence gate abstains before scoring when
either required stream is absent.

Report on development validation: Accuracy, NLL, Brier, 15-bin equal-width
ECE, mean confidence, confusion matrix, train-versus-calibration gap, exact
sample predictions/probabilities, consumed/unique rows and sources, wall time,
evaluation time, peak VRAM, and parameter/readout bytes. Report both variants
and every control; select using validation only if a later explicitly scoped
development diagnostic is needed, and do not call that blind evaluation.

For uncertainty, perform 2,000 paired bootstrap replicates with seed 17,
sampling the 1,374 validation pair groups with replacement and preserving all
eight rows per sampled group. Report percentile 95% intervals for each
variant and for `gru - mean` differences in Accuracy and NLL. This is a
single-seed development comparison; no extra seeds are authorized in this
protocol.

## Interpretation boundaries

A positive joint result would support only cross-clip AVMIT action-label
matching from these two precomputed feature sources. It would not establish
general audio/video reasoning, temporal causality, raw-media runtime quality,
Path A compatibility, four-modality performance, or product deployment fit.
A null result is retained as-is and does not prove a universal model-family
limit. No sealed-audit or published test result may be reported.

## Pre-fit pair-construction correction — 2026-11-10

The initial implementation generated `pairs-v2` with only two distinct
cross-label source combinations per group (each duplicated in reverse), while
the protocol required all four cross-label combinations. This was found during
the pre-training audit. A prior v2 smoke used eight **training** groups for
forward/backward, but performed no optimizer update, calibration selection, or
development-validation inference. Full fitting had not begun. The feature
cache had been decoded against v2; v3 was verified to have identical source
asset sets, so no features needed re-extraction. The v2 manifests, cache and
smoke report are retained unchanged as history.

The corrected immutable manifests are `split-v1/pairs-v3/` outside Git. The
four negatives now form a cycle through every cross-label edge exactly once;
the four positive rows still contain the two same-label edges in both
directions. All four clips remain represented once under each target as audio
and once under each target as video. Tests check the full cross-product edge
set, and the training input validator independently enforces it.

| Split | Groups | Rows | Unique clips | pairs-v3 SHA-256 |
|---|---:|---:|---:|---|
| Train | 9,552 | 76,416 | 38,208 | `d003a693668c34d4da31d242295511e8836945e61780810df6db23ded88488b7` |
| Calibration | 1,317 | 10,536 | 5,268 | `c0f857319413121764c9fb2f5eb1458bb5ce73027a049a8c146df90160dd36ed` |
| Development validation | 1,374 | 10,992 | 5,496 | `e473fd1c9c5527a6ea949440eb991732fe996d3dd04ac39541d905288b7617eb` |

The pair-v3 asset identity sets exactly match the corresponding v2 and feature
cache identities in all three allowed splits. No features were re-extracted.
For the new run, `features-pairs-v3/` is a separate cache view containing
hardlinks to the already verified v1 feature arrays/indexes and a new report
bound to the v3 pair hashes. The original `features-v1/cache-report.json`
SHA-256 remains `4b88888b09676c854a6272edb136ef91b3d2240468a0a596a80f6b203d6f8e7f`;
the new report SHA-256 is
`71533d86b62a7eedb45586f90684ce96a7bfa0e5b9875822e2bb8a347d3f179b`.
The new cache report continues to record zero sealed payloads and zero raw
media opened. All subsequent smoke, fit, and validation commands must point
only to `pairs-v3/` and `features-pairs-v3/`.

## Fixed-seed fit and development validation — 2026-11-10

**Scope:** This completes the one predeclared seed-17 AVMIT feature-readout
comparison only. It does not complete the broader pretrained-representation
roadmap or qualify a product model.

- Source code commit: `e5d8f0a597aa7af2a31ed44a7259ae7214d61a5b`.
- Exact command:
  `python scripts/run_avmit_savc_experiment.py train --cache D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01\features-pairs-v3 --pairs D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01\split-v1\pairs-v3 --output D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01\run-v1\fit-v1 --device auto --smoke-report D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01\run-v1\smoke-v3\smoke-report.json`
- Initialization: two fresh seed-17 readout initializations; no checkpoint
  warm-start. The mean and GRU share identical data, scorer, optimizer,
  schedule, calibration selector, and fixed train/calibration/validation IDs.
- Environment: RTX 3080 Laptop GPU, 16 GiB; Python 3.11.9, PyTorch
  2.6.0+cu124, CUDA runtime 12.4; FP32 readouts over cached FP32 features.
- Process created at 2026-11-10 13:40:19 JST; results file was written at
  13:41:30 JST. Runner wall time was 63.172 seconds; recorded training time
  was 22.983 seconds for mean and 19.471 seconds for GRU. Calibration
  evaluation took 3.093 and 2.513 seconds; final validation took 0.562 and
  0.547 seconds, respectively. No cloud/paid compute was used.
- Effective config SHA-256:
  `d05d2810eae66c1fcedc3079ec317294e4c8fa2dd53837b75e76d3340b85231d`.
  The exact serialized config and environment are in the results artifact.
- Run-wide data: 76,416 unique training rows, 38,208 unique source clips and
  9,552 groups. Mean consumed 458,496 rows (6 complete passes; 382,080
  repeated row-consumptions); GRU consumed 382,080 rows (5 passes; 305,664
  repeated row-consumptions). Calibration: 10,536 rows / 1,317 groups / 5,268
  clips. Development validation: 10,992 rows / 1,374 groups / 5,496 clips.
  Train source clips are disjoint from calibration and validation by the
  frozen asset IDs. No sealed or publisher-test records were used.
- Peak allocated VRAM: mean 92,640,256 bytes; GRU 144,313,344 bytes. Best
  readout parameter counts and sizes: mean 180,865 parameters / 726,108 bytes;
  GRU 500,993 / 2,007,490 bytes. These counts exclude upstream feature
  producers and are not a complete deployed-model/package measurement.
- The selector chose mean epoch 2 and GRU epoch 1 by lowest calibration NLL.
  Training stopped by the fixed four-epoch patience rule, not by an adjusted
  quality threshold. Best and final checkpoints are both preserved.

### Learning curves and overfit signal

| Readout | Epoch | Train CE | Calibration NLL | Selected |
|---|---:|---:|---:|:---:|
| Mean | 1 | 0.60212 | 0.58934 | |
| Mean | 2 | 0.51702 | 0.58873 | yes |
| Mean | 3 | 0.48104 | 0.59054 | |
| Mean | 4 | 0.44952 | 0.60349 | |
| Mean | 5 | 0.41645 | 0.63672 | |
| Mean | 6 | 0.38480 | 0.68829 | |
| GRU | 1 | 0.61938 | 0.59093 | yes |
| GRU | 2 | 0.52158 | 0.59314 | |
| GRU | 3 | 0.46717 | 0.59672 | |
| GRU | 4 | 0.41595 | 0.62474 | |
| GRU | 5 | 0.35821 | 0.70440 | |

Both training losses fell while calibration NLL rose after the selected
checkpoint. This is direct evidence of overfitting within this fixed recipe.
No later epoch was selected because its training loss was lower.

### Development validation metrics

All rows below are the single post-freeze pass over the fixed 10,992-row
validation set. Temperature was fitted only on the 10,536 calibration rows.

| Readout | Calibration temperature | Validation calibration | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---|---:|---:|---:|---:|---:|
| Mean | 1.37466 | unscaled | 0.67194 | 0.60108 | 0.41371 | 0.06603 | 0.73798 |
| Mean | 1.37466 | scaled | 0.67194 | 0.58665 | 0.40467 | 0.02322 | 0.69225 |
| GRU | 1.11806 | unscaled | 0.66148 | 0.60272 | 0.41811 | 0.04575 | 0.70427 |
| GRU | 1.11806 | scaled | 0.66148 | 0.60003 | 0.41576 | 0.03523 | 0.68794 |
| Structural unimodal control / chance | — | — | 0.50000 | 0.69315 | 0.50000 | 0.00000 | 0.50000 |

The fixed readouts exceed the structural 50% one-source ceiling on this narrow
label-matching task, but remain far below the project's broad multimodal
quality aspiration. Mean pooling is ahead of GRU by 1.046 percentage points
of accuracy. The 2,000-replicate paired group bootstrap (seed 17) gives a
95% interval of `[-1.865, -0.264]` percentage points for `GRU - mean`
accuracy; for NLL the difference is `+0.00164` with interval
`[-0.00438, +0.00779]`. Thus this one validation comparison favors mean
pooling on accuracy, while NLL does not resolve a difference. It does not
establish a general temporal-architecture result.

An independent fresh-model reload of each selected `best.pt` reproduced the
saved validation row IDs, targets and logits (absolute tolerance `1e-6`) for
all 10,992 rows; recomputed accuracy/NLL/Brier/ECE exactly matched the saved
uncalibrated validation metrics. Reloaded checkpoint SHA-256 values:

- Mean best: `a3b6592ad06f94bec44df361773ae4790c9b8686a035133eb44075f09974ea3f`.
- GRU best: `ef9dd0bbf89ba3fd83fb982a8fe850e55bd419d264b48bdffb910981e78430f2`.
- `results.json`: `295e57cc66cceb7ec422269396c37c348b4c071d9d76b9773097ed5c62cbc4d3`.
- Full 21-file hash inventory (smoke, fit, predictions, v3 pair manifests and
  cache report): `run-v1/artifact-hashes.json`, SHA-256
  `d7f1ba838df7b5ef58dcb9a542a185052cfac61852b6622ee67099e82e1a1815`.

The complete `fit-v1/` directory also contains final checkpoints, full
per-epoch histories, and exact calibration/validation predictions with IDs,
targets, logits and probabilities. It is outside Git at
`D:\CodexArtifacts\pretrained-reuse-r2\avmit-metadata-gate-20261110-attempt-01\run-v1\fit-v1`.
No weights, features or media are added to Git. The sealed audit was not
opened. No additional seed, hyperparameter change, or product promotion
followed this result. Continue R2 only with a distinct, predeclared task/source
if roadmap gates justify it; do not tune this validation set further.
