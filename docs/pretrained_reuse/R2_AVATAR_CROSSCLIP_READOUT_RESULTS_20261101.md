# AVATAR Cross-Clip Audio-to-Video Decision Readout (2026-11-01)

> **Superseded:** The initial corpus/readout described below had a target
> construction shortcut: the positive video frame was the query audio's own
> asset at the exact same timestamp. Its metrics are retained only as a
> diagnostic and are not valid AV quality evidence. The full audit, corrected
> cross-source construction, and separate readout are documented in
> [the 2026-11-02 report](R2_AVATAR_CROSS_SOURCE_READOUT_20261102.md).

## Status

**Feature extraction is in progress; no readout has been fit and no quality
result is available yet.** This is a development-only 4-choice category-match
adaptation. It is not AVATAR's native spatial sound-localization task and is
not evidence of general audiovisual or four-modality Decision quality.

The held-out audit split and its annotations/media have not been opened or
loaded. No product promotion is implied.

## Frozen data preflight

The task inputs are the frozen files under
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\option-choices-04`.
The train and validation sets contain 11,995 and 1,693 four-choice examples.
An independent full-file consistency pass checked every model-input row
against its analysis sidecar, including option order, target index, target
category, and target source group. It found:

- zero example-ID overlap;
- zero overlap among target and negative source-video groups;
- zero audio/video media-identity overlap;
- zero normalized input-content overlap;
- all 80 train categories represented in train; 71 categories occur in
  validation, all seen in train. Validation category support ranges from 1 to
  86 examples, so category-macro metrics must be reported with support counts.

Target option positions are near-balanced: train `[2961, 2950, 3012, 3072]`,
validation `[432, 426, 423, 412]`. The candidate options are ordered by a
deterministic hash and use opaque `option_0` through `option_3` IDs.

The EfficientAT audio cache was independently reopened and checked against the
frozen train/validation input IDs. It contains exactly 13,688 unique finite
384-D features, with no missing or extra keys. Its SHA-256 is
`018bd96b433c5a56d26a6d26c101a5a5ff1a399c7e891efdcf5a681fc423c32c`.
The runtime gate matches every cache center and declared padding to the frozen
inputs. Actual stream padding exceeds the declaration for 738 windows (maximum
3,248 samples / 101.5 ms); none is shorter than declared. This reflects actual
audio stream duration and is retained as a limitation/control, not silently
rewritten in the corpus.
The full row/sidecar and audio-cache preflight receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-03\readout-preflight-01\preflight.json`
(SHA-256 `39d09f63f23fae0d132407d7e50f0b1dae6eba163d38467fa174791f54929064`).

The VideoPrism feature extractor is processing the corresponding 13,688
candidate assets on CPU in the separate external output directory
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-03\video-features-full-01`.
At the last verified check it was RUNNING at 1,640 / 13,688 assets; process
PID 26652 was alive. The extractor's per-shard hashes and progress remain
outside Git. No GPU training, cloud compute, audit access, or changes to legacy
runs occurred during this preflight.

## Planned readout comparison

After both frozen caches pass exact ID, shape, finite-value, and hash checks,
the readout script will compare, on the same validation IDs:

- uniform chance and a train-derived option-position prior;
- an audio-only category classifier. Its offline candidate-category lookup
  uses sidecar labels, so it is an oracle diagnostic and is not a deployable
  option predictor;
- metadata-only, video-only, and joint audio/video readouts;
- joint readouts with either stream removed, with validation audio shuffled
  across distinct audio asset IDs, and with video option order shuffled.

The learned readouts use seed 17, an AdamW optimizer, at most 50 epochs,
validation-NLL selection, patience 6, and `min_delta=0.001`. Train/validation
NLL gaps, macro-by-category and per-category metrics, per-target-source-group
metrics, paired target-scene bootstrap intervals, predictions, checkpoint
hashes, parameter counts, runtime, and peak CUDA allocation will be retained.
The metadata and category sidecar are not used as joint-model feature inputs.
The runtime also rejects audio center/padding or video timestamp mismatches
against the frozen input files, and records measured stream-padding and video
frame-seek error distributions before fitting.

## Results

Pending completion of VideoPrism extraction and the fixed readout comparison.
