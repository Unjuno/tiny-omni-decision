# Text-free native-feature Decision diagnostic (2026-10-31)

## Scope

This is one fixed-budget implementation diagnostic of
`NativeFeatureDecisionScorer` on the existing eight-taxon iNaturalist image
task. It is not an open-vocabulary Decision benchmark, a blind evaluation, a
multimodal-fusion result, or evidence for the 95–98% product target. The same
430 observer-disjoint validation records had previously been used to select a
different readout. This run did not fit on validation or select/stop by
validation; its comparison is still post-selection development evidence.

The upstream iNaturalist observations and photos were explicitly CC0. The
experiment consumed only the frozen feature cache; no media, source API,
sealed audit, or encoder weights were opened. VideoPrism-B stayed frozen.

## Inputs and integrity

The assembled cache has 1,575 rows: 1,145 train and 430 validation, all eight
taxa in both partitions. Before fitting, the runner checked SHA-256 for the
feature matrix, feature-row manifest, and prior validation predictions. It
also checked unique sample IDs and zero train/validation overlap for observer
group, observation ID, and photo ID. The source manifest reports zero exact
image and pHash-distance-4 cross-split overlaps. No source labels, taxon names,
photo IDs, or observer IDs were provided to the scorer; the only candidate
semantics were fixed one-hot taxon IDs in a task-owned `TypedOptionSchema`.

Frozen feature provenance: VideoPrism-B revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, source revision
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
The encoder has 114,365,184 parameters and was not loaded for this fit.

## Fixed run

- Source code base: `0a592696f3b4ac2c9197cba6d2ef771a4ba733f4`.
- Seed 17, CPU, 10 epochs, 18 optimizer updates per epoch (180 updates total).
- `NativeFeatureDecisionScorer(768, 8, hidden_width=64)`, 49,793 trainable
  parameters; AdamW, learning rate 0.003, weight decay 0.0001, batch 64.
  Including the frozen VideoPrism encoder, the exercised image path contains
  114,414,977 parameters before quantization.
- Feature standardization used train-only mean and population standard deviation
  with a `1e-6` floor. No encoder or projector was trained.
- The final epoch was retained; there was no validation checkpoint selection,
  early stopping, calibration fit, or hyperparameter search.
- CPU fit/evaluation wall time: 2.539 seconds. No GPU or cloud compute was used.
- Output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\native-feature-decision-inat-20261031-attempt-01`.
- Config SHA-256 `0a47172061639ae11881313dd7a941d9d7557d120374eb0539f1ea74cedfd96c`;
  final checkpoint SHA-256
  `dd4178b5ae1945655c48f61c92f60cebfc4da29c9fda11a5d75e25c38360132d`;
  history SHA-256
  `471f975bbd94ab8743d09dc0f8a6f07c4a0e8b85531cfbdafbd51bdc51c1b18a`;
  validation predictions SHA-256
  `aef976e2679b5d366752e1f552ba30e73124a7a1c62f960fa289f1ac72ff43ef`.
- Executed fit-script SHA-256
  `d424b644fa3716b52fda9f8981984981199ac01c746f54b4d2a4bbbe39f1311a`;
  comparison-script SHA-256
  `7bc2ed63c52f8f19135176c1c831cd1fd905b4b2499645c3703c0edaa62bafd8`;
  replay-script SHA-256
  `9ac83a45e40e312f0b6bb2ff45d9d037849d178fdc32f40c0e5bac2a4d300351`.
- The exact executed fit and comparison scripts, command/environment receipt,
  config, history, predictions, checkpoint, and result remain together in the
  output directory. The committed scripts include lint-only line wrapping after
  this run; the byte-exact executed copies are retained alongside the run.
- A separate CPU process reloaded the saved final checkpoint and replayed all
  430 validation records. Option predictions were identical; maximum absolute
  probability delta was `7.15e-7`. Replay receipt SHA-256 is
  `b6a6f9b39515f54ef1197db4e3cf8d9b95dd367a4efc91bee188956cc780a134`.

## Results

The prior affine baseline and the new scorer were evaluated on the exact same
430 validation IDs and order. Metrics were independently recomputed from the
saved probabilities. Differences below are new minus baseline.

| Metric | Affine baseline | Native-feature scorer | Difference | 95% paired observer-cluster interval |
|---|---:|---:|---:|---:|
| Accuracy | 0.7488 | 0.7535 | +0.0047 | [-0.0301, +0.0385] |
| Macro Accuracy | 0.7190 | 0.7253 | +0.0063 | [-0.0418, +0.0536] |
| NLL | 1.1258 | 0.7093 | -0.4166 | [-0.6086, -0.2441] |
| Brier | 0.4004 | 0.3509 | -0.0495 | [-0.0921, -0.0068] |
| ECE (15 bins) | 0.1562 | 0.0502 | -0.1060 | [-0.1302, -0.0447] |
| Mean confidence | 0.9050 | 0.7522 | -0.1528 | descriptive |

Native-feature per-taxon validation Accuracy / NLL:

| Taxon ID | N | Accuracy | NLL |
|---:|---:|---:|---:|
| 12727 | 53 | 0.5849 | 1.1667 |
| 13858 | 74 | 0.8919 | 0.3212 |
| 144455 | 80 | 0.8125 | 0.5340 |
| 3017 | 51 | 0.8235 | 0.6453 |
| 4956 | 49 | 0.6327 | 0.8474 |
| 6930 | 53 | 0.7170 | 0.8814 |
| 7089 | 54 | 0.7778 | 0.7265 |
| 9083 | 16 | 0.5625 | 1.0179 |

Intervals use 10,000 paired bootstrap draws, seed 17, resampling the 300
observer groups and retaining all rows for each sampled group. Accuracy and
macro-accuracy differences remain uncertain; the saved probabilities show
lower NLL, Brier, ECE, and mean confidence for this fixed diagnostic. That is a
calibration/overconfidence improvement on these development IDs, not evidence
of better general Decision accuracy. The run reached 98.17% train Accuracy and
75.35% validation Accuracy at epoch 10, a 22.82-point gap. The training curve
continued upward while the development result remained far from the product
target.

## Decision and remaining gates

The typed-option implementation can fit, score supplied candidates, preserve
probability normalization, and produce a reproducible fixed-feature evaluation
with fewer than 50k trainable parameters. On this narrow task the point Accuracy
gain is only 0.47 percentage points and its paired interval includes zero;
loss and calibration metrics improve, but the validation was already involved
in earlier model selection. Do not promote this readout or reuse the same
validation for further architecture tuning. A fresh, larger observer-disjoint
cohort and task-specific Decision evaluation are required for a quality claim.

This closes only a first measured M1 exercise for the new text-free typed
readout. Full M0/M1 remains open across modalities and task types; R2-M joint
input necessity remains open; R3–R8 are not unlocked by this diagnostic. The
sealed audit, ternary export, quantization, merging, and product promotion were
not performed.

## Runtime adapter normalization follow-up

After the diagnostic was recorded, an implementation-path mismatch was found: the scorer was fitted on train-only standardized image features, but the hash-checked cached-feature JSON adapter had no way to apply the saved training mean and scale. This could make cached inference use a different feature scale from the fitted readout. The adapter now accepts an optional mean/scale pair, validates that both are present, finite, and match the feature width (with strictly positive scale), and normalizes gathered features before scoring. Existing callers remain unchanged when neither value is provided.

Regression tests verify parity with a direct standardized readout and reject partial, wrong-width, and non-positive normalization. The real cache-to-JSON path was replayed on the same 430 fixed development image IDs in a fresh external output directory. Every argmax prediction matched the saved direct readout; maximum probability difference was `9.54e-7`. Cache-write/read/decision time averaged `0.00199 s/example` on CPU. The run receipt SHA-256 is `E7B534B2DDFAA12D38378DAEC8D1DB2EC0EAA693C1F5955E84ECE5D8EEEF5233`; prediction SHA-256 is `16d654caa88ec9338cca6853cdded3dc3324717eee6e63184366c85b4f873fd7`. Receipt and per-example JSON remain outside Git under `C:\CodexArtifacts\pretrained-reuse-r2\native-feature-decision-inat-20261031-attempt-01\runtime-json-replay`.

This is an integration-parity check on already observed, post-selection development IDs; it is not a new quality evaluation and does not change the accuracy conclusion above. It does not unlock another modality, joint-input claims, sealed-audit evaluation, quantization, or product promotion.
