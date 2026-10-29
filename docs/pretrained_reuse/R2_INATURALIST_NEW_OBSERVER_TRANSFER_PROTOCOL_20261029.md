# iNaturalist new-observer transfer protocol (2026-10-29)

## Question and scope

Measure once whether the already-selected VideoPrism-B frozen image readout
transfers to the 176 new observer-disjoint validation records in the
2026-10-29 CC0 expansion. This is an external development-transfer diagnostic
for the same narrow eight-species image-classification task. It does not train
or select a model and is not a sealed audit, an open-vocabulary Decision task,
or evidence for the four-modality product target.

The previous 700-train / 200-validation expansion-training gate remains
failed. This separate evaluation uses only the pre-frozen new validation
records; it does not relax that gate, reuse the old validation split, or add
the new records to training.

## Frozen model and data

- Readout checkpoint: `C:\CodexArtifacts\inat-birds-videoprism-readout-v1-20261026-attempt-02\best-readout.pt`
- Readout SHA-256: `323d0954a83c037ba5c6b37da3997addba48a66787a072acd3bae5fdbe7d2b88`
- Original feature source: VideoPrism-B revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`.
- VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- New validation manifest SHA-256:
  `d5028e54f7d0000c0936f7319f6f8a598f6c458be02f10b2555ea7baf74729b4`.
- Fixed new validation count: 176 across eight taxa. Do not remove or replace
  rows based on content or predictions.
- Both observation and selected-photo license fields must equal `cc0`.
- Use the HTTPS-only, host-checked, 8 MiB-bounded downloader and the same
  EXIF-oriented RGB, 288 short-side/center-crop, repeated 8-frame preprocessing
  and mean-token 768-vector feature pooling. One frozen row uses a `.png` source
  URL; accept it only when the same-host HTTPS endpoint responds with `image/png`.
  Retain strict redirect, size, decode, and hashing checks for both formats.
- Feature extraction uses the pinned native JAX CPU worker. No CUDA/GPU model
  loading, checkpoint modification, training, calibration, or tuning.

## Integrity gates before scoring

1. Hash the complete validation manifest and verify its 176 IDs/order against
   the frozen expansion metadata.
2. Download only these 176 validation assets. A failed or mismatched asset
   makes the full evaluation incomplete; do not silently drop it.
3. Verify every raw download hash, normalized RGB hash, photo ID, and CC0 field.
4. Compare new validation raw/media/RGB hashes and 64-bit perceptual hashes
   against all 1,642 old train images and 683 old validation images. Require no
   exact overlap and no pHash Hamming distance <=4. On any such collision,
   stop before producing quality metrics; do not selectively filter examples.
5. Verify feature finiteness, sample order, checkpoint hashes, and identical
   class-ID order before loading the frozen readout.

## One-shot metrics and controls

On the full fixed cohort, report overall and macro-over-eight-taxa Accuracy,
NLL, multiclass Brier, 15-bin ECE, and mean confidence, plus each taxon's
sample count and metrics. Save one prediction record per sample containing
the sample/photo identity hashes, target, prediction, and the full ordered
probability vector. Report missingness or integrity failure without scoring
if a gate fails.

Alongside the aligned prediction, compute two fixed sensitivity controls with
the same frozen readout: all-zero image features and a deterministic
derangement of image features by SHA-256-sorted sample ID (no fixed points).
These are diagnostics only; they cannot select or calibrate a model. Report
paired observer-cluster bootstrap intervals for aligned-minus-control
Accuracy and NLL with 10,000 resamples and a fixed RNG seed, along with the
number of independent observer groups. These intervals describe this
development cohort and do not account for checkpoint selection on the older
validation split.

No further evaluation of this 176-record cohort will follow this one-shot
report. No sealed audit is opened or created. Preserve all 2026-10-26 source,
feature, readout, and validation artifacts unchanged.

## Resource preflight and stop conditions

The local free space before the pilot was 395,751,424 bytes. The first three
sorted validation assets downloaded successfully in 4.843 sec and totaled
647,109 raw bytes / 1,035,616 normalized bytes: means 215,703 raw bytes and
345,205 normalized bytes per image. Full acquisition then completed for all
176 rows in 242.0 sec (27,128,015 raw bytes; 60,046,213 stored bytes), leaving
393,146,368 bytes free. Feature extraction took 223.1 sec on the pinned CPU
worker (3.17 sec model load). The PNG row was accepted only after the same-host
HTTPS endpoint returned `image/png`; all strict host, redirect, size, decode,
and hashing gates were retained. The measured storage margin exceeded 100 MB.
See `R2_INATURALIST_NEW_OBSERVER_TRANSFER_RESULT_20261029.md`.

If storage projection, network access, source/media integrity, or old-cohort
overlap checks fail, stop and record the exact reason. Do not switch to a
different validation subset or retry using the old evaluation.
