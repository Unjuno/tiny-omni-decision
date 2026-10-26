# STARSS23 spatial-option Decision task: training-label preflight

## Why this task

The existing STARSS23 event-presence task asks whether a class is active in a
five-second interval. Its AV readout did not beat audio-only, and it cannot show
that video helps locate an event. STARSS23 version 1.1 supplies synchronized
360-degree video, four-channel audio, and time-indexed direction annotations;
the official AV SELD task is a spatial localization task. The released record
declares MIT and the source probe retains its license and README hashes. This
supports a bounded, local research comparison, not a product-quality claim.

## Proposed Decision interface

For a fixed five-second synchronized observation and a caller-supplied event
class plus candidate azimuth sector, answer whether an annotated source of that
class occurred in that sector during the interval:

```text
question: Is <event class> present in the <direction> sector in this interval?
options: ["no", "yes"]
target: 0 or 1
```

Use eight 45-degree sectors centered at front, front-left, left, back-left,
back, back-right, right and front-right. The STARSS paper defines azimuth zero
as front and +90 degrees as left; it says the labels were checked by overlaying
their corresponding azimuth/elevation markers on the equirectangular video.
The DCASE 2023 technical report describes the panorama horizontal axis as
azimuth +180 degrees at the left through 0 degrees at the center to -180 at the
right. Together these imply that the 16 VideoPrism patch columns map as follows:

| Decision sector | Patch columns |
| --- | --- |
| front | 7, 8 |
| front-left | 5, 6 |
| left | 3, 4 |
| back-left | 1, 2 |
| back | 15, 0 (panorama seam) |
| back-right | 13, 14 |
| right | 11, 12 |
| front-right | 9, 10 |

The camera is mounted 10 cm above the microphone array. This supports the
coordinate mapping; it does not establish that every annotated source is
visually detectable. Each five-second half-open interval `[start, end)` uses
100-ms annotation ticks and creates 96 binary questions (12 event classes
after merging female/male speech, times eight sectors). For each event and
interval, a positive answer means at least one annotation for that class falls
in that sector. This preserves simultaneous sources in different sectors
without forcing a single-winner label. Retain all classes and report classes
with zero training support.

Sources: [STARSS23 paper, dataset geometry and annotation convention](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6c9671ed3b3106b71cafda3ba225c1a-Paper-Datasets_and_Benchmarks.pdf),
[DCASE 2023 technical report describing the panorama azimuth axis](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Du_102_t3.pdf).

## Representation and controls

- Audio: retain all four FOA channels. The existing channel-0-only audio cache
  is not a spatial-audio representation.
- Video: use the pinned VideoPrism-B checkpoint and retain direction-indexed
  spatial patch features. Its source documents outputs shaped
  `[batch, 16, 16, 16, 768]` after reshaping: time, patch row, patch column,
  feature. The current STARSS extractor averages all tokens and discards
  spatial position, so it cannot answer this task. Any new extractor must keep
  the full equirectangular horizontal coordinate; center-cropping the panorama
  is disallowed.
- Compare prior-only, audio-only, video-only, joint, video-zeroed and
  room/recording-shuffled-video controls on identical examples and IDs.
- Split by room first, then recording. Every interval, event query and
  direction choice inherits its recording's room split.
- Report Accuracy, balanced Accuracy, NLL, Brier and ECE by event class and
  sector, plus macro scores and room-grouped uncertainty. Keep every
  direction-query at evaluation; do not report raw Accuracy without the
  prevalence baseline.

## Executed label-density preflight

The CPU runner
`scripts/preflight_starss23_spatial_decision.py` reads only the pinned
49-recording `dev-train` annotation manifest and the corresponding members of
`metadata_dev.zip`. It does not read media, calibration/validation annotations,
`dev-test` annotations or the official unlabeled evaluation release. It performs
no training or model inference.

The exact split-manifest SHA-256 is
`571c2b537a809b2f92c9289fd932c20e0169ca1c0e38f4dc0f4c6ad6ef2ac978`; the
metadata archive SHA-256 is
`2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6`. Results:

- 49 training recordings across six rooms, with 6–11 recordings per room.
- 116,750 annotation rows after speech-label merge and 102,434 unique
  `(recording, 100-ms tick, event class)` groups. These correlated groups are
  not independent examples.
- Same-class simultaneous source rows occur in 5,553 speech groups and 6,649
  music groups. Multiple sectors occur in 4,838 speech groups, 2,811 music
  groups, 968 musical-instrument groups, and fewer groups for other classes.
  This supports a per-sector binary target over a single categorical target.
- Bell has zero training annotations. Clapping has 151 rows and knock has 58;
  these support limits must remain visible in any model result.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/`
`starss23-spatial-decision-preflight-02/result.json`.
The runner SHA-256 is
`3a51924fd970ffafc263bd374ef3bbcc6e9ed77b0183234c3ca2c1a865a063fd`.

## Executed one-window coordinate and target gate

The pinned dev-train smoke now checks all eight selected video presentation
times against the paired five-second audio interval and constructs the 96
event-sector labels from only that recording's pinned dev-train annotation
member. The code uses the column table above, including wraparound at the
panorama seam. A separate CPU VideoPrism replay checks the `[8,16,16,768]`
token grid and exact parity of the global pooled embedding with the existing
frozen cache; it does not read labels or use the GPU. These are data/geometry
checks, not visual-source detectability or quality results.

- Training-window gate: `starss23-spatial-training-gate-01/result.json`,
  SHA-256 `59295e2ae3e7b19a1cb8a379ef9f87d8fb8775a4045638506a85ed31a262cf81`;
  96 targets with one positive (`speech/back`) in the
  selected `[0, 5)` second window. The eight frame presentation timestamps
  range from 0.333667 to 4.704705 seconds.
- Spatial-token smoke: `starss23-spatial-token-smoke-02/result.json`,
  SHA-256 `821e4bdf3e91df8b02d9d80414b02d1c0d663de710364845f91bfd633b7920b`;
  CPU only, 11.848 seconds, output `[1, 8, 768]`,
  existing pooled-feature maximum absolute difference `0.0`, feature file
  SHA-256 `e6c4ca763a986719d5ec9015419a35899dbe05c756795c074802e973f9c9b5f6`
  (24,704 bytes). The separate system Python lacked JAX; the already-pinned
  VideoPrism environment succeeded without package changes.

## Frozen development corpus and imbalance check

The pinned room split contains 78 dev-train recordings: 49 train, 8
calibration, and 21 validation recordings, all assigned by room before
windowing. The new corpus builder generated 96 binary questions per five-second
window, for 187,680 train, 26,784 calibration, and 46,656 validation queries.
Output: `starss23-spatial-option-corpus-v1/result.json`, SHA-256
`7df56dd1dc472b8668b6b7c5cd7eb684e405771f88f4b8026049656610c61d5a`.
The train/calibration/validation JSONL hashes are recorded in that result.
No official `dev-test`, evaluation, or sealed-audit labels were read.

There are only 4,203 positive train queries (2.2395%), 458 calibration
positives (1.7100%), and 1,258 validation positives (2.6963%). An always-`no`
predictor therefore gets 97.7605% / 98.2900% / 97.3037% raw Accuracy on these
three splits. Raw Accuracy alone is not a useful quality criterion here; the
readout must report balanced Accuracy, NLL, Brier, ECE, per-event and per-sector
metrics, and the always-`no` baseline. Bell has zero training positives but
32 validation positives, so it is an explicit unsupported-class transfer test,
not evidence that the training split covers all events.

## Frozen-feature extraction and Decision runner status

The full audio feature extraction completed in a new artifact directory:
`starss23-foa-channel-corpus-features-01/result.json` reports `PASS`, 2,720
windows, 78 recordings, `[2720,4,384]` features, 843.37 CPU seconds, and a
maximum channel-0 difference of exactly zero against the existing pinned mono
cache. Feature matrix SHA-256 is
`0d6aee00ded4db333f0e38b1d7cbee28cc70171683a4f487563cd102a17f9729`. The
isolated `torch/torchaudio 2.6.0+cu124` environment was used; no global or
existing VideoPrism environment was changed. This does not claim that an
AudioSet mono model is a spatial-audio encoder; it tests whether independently
encoded FOA channels retain useful direction cues.

Frozen VideoPrism spatial extraction is still running on CPU in
`starss23-spatial-corpus-features-01`. At the latest recorded checkpoint it
had completed 1,460 of 2,720 windows (39 recordings); the runner saves and
hashes one shard per recording and reads no labels. No GPU training has begun.

The new `scripts/train_starss23_spatial_decision.py` implements the next
development-only Path A comparison: frozen audio-only, spatial-video-only, and
joint features; event and sector conditioning; supplied `[no, yes]` options;
balanced positive/negative training loss; train-window-only feature
normalization; calibration and checkpoint selection on the independent
calibration rooms; and final reporting on the two held-out validation rooms.
Checkpoint selection and calibration use macro event-sector NLL, not
sample-weighted Accuracy. Reports include overall and per-event/per-sector/
event-sector metrics, per-room results, trainable head size, saved predictions,
and audio/video ablation plus cross-recording video permutation controls. A
CPU synthetic train/reload smoke and feature-identity checks pass. The three
readout heads have 199,681 (audio), 789,505 (video), and 986,113 (joint)
trainable parameters at hidden width 128. Per-dimension normalization is fit
only on unique training windows. The video permutation control preserves room
identity while changing recording identity. The targeted suite passes 41
tests. This is pipeline verification only; there are no measured STARSS
readout metrics yet. GitHub Actions for commit
`9572cd77ba94c8b7e91a3ad6cd55934efa424014` passed lint, all CPU tests, and
manifest validation (run
[`40295306030`](https://github.com/Unjuno/tiny-omni-decision/actions/runs/40295306030)).

## Next gate and stop rules

Before fitting, finish VideoPrism extraction, confirm its manifest order and
media hashes against the frozen audio features and corpus, and verify free disk
and local runtime. The runner must reject wrong row order, split, or media
identity. Reuse no official dev-test or sealed-audit labels.

If the smoke passes and local resources allow, fit one frozen-feature readout
with a fixed room-grouped protocol, then evaluate the untouched official
`dev-test` recordings once after all choices are frozen. This is a development
transfer evaluation, not a sealed audit. A joint score only supports
complementary AV use if it beats both matched unimodal controls and degrades
under video ablation or recording/room permutation; sensitivity alone is not
enough. The existing event-presence result remains unchanged. No shared-Lens,
backbone adaptation, quantization or product promotion follows automatically.
