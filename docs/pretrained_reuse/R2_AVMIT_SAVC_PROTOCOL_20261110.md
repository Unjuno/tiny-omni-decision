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
