# AVATAR audio-video entity Decision protocol (2026-11-01)

## Status and source lock

This is a bounded R2 development protocol for a new audiovisual task. AVATAR
is admitted for local development with attribution under the pinned CC BY 4.0
dataset declaration; this does not grant any right beyond the stated dataset
and upstream media terms. Do not redistribute source media. The official
VGGSound source page also states CC BY 4.0 and notes that source-video
copyright remains with original owners.

Pinned dataset: `mipal/AVATAR@a7f0146ff704799d044133e61137836cacc1c233`.
The immutable metadata lists 5,000 videos, 24,266 annotated frames, and
28,516 instances across single-sound, mixed-sound, multi-entity, and off-screen
scenarios, with 80 audio-visual categories. `video.zip` is 3,785,157,586 bytes
(SHA-256 `5ce5be7fa6dda85889750f6c7cc2fad807527063e7d7090c664170a66b078a68`);
`metadata.zip` is 1,624,245,410 bytes (SHA-256
`97e9f09e9f4325ac888db2886da22ca4650eafb8848721230a4e5d80fb6a4183`).
The video archive was downloaded to
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\video.zip`
and its length and SHA-256 were verified. The metadata archive was downloaded
to the same attempt directory and verified against its pinned length and
SHA-256. Only train and validation annotation JSON members were opened and
extracted; no JPG previews were extracted, and no audit annotation member was
opened. A later bounded media-container pass is described below. The extraction receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\annotation-extraction-02-receipt.json`.
It records 21,682 extracted JSONs (242,943,743 bytes), extracted-file manifest
SHA-256 `c8125861e0b63ca25df8bb8b21e418701a2f0b5c64a1970d22ec09f9c3581a44`,
and 2,584 withheld audit JSON members.

The public release does not expose an AVATAR train/validation/test split.
The video ZIP contains 5,000 actual clips and 5,000 `__MACOSX` sidecars. The
actual clip filenames map to 4,108 source YouTube IDs; 568 IDs have multiple
clips and one source has up to 24 clips. Therefore derive a deterministic split
from source YouTube IDs only: seed 17, 80% train / 10% validation / 10%
held-out audit, with every clip, frame, and instance from one source ID kept
together. The ZIP-member-only splitter is
[`scripts/prepare_avatar_group_split.py`](../../scripts/prepare_avatar_group_split.py).
It produced 3,286 train / 411 validation / 411 audit source groups containing
3,952 / 520 / 528 clips respectively. The split artifact SHA-256 is
`57766c748e2d567060685cc16d8be554201f602224b81e6b3ee13f61d987b9af`; its
canonical manifest hash is `bd01932a05952e303b9d0d97c2c15870d8c4315e3e50dc9b08db7e4e3b4cdc41`.
This is a project-defined split of a public benchmark, not a blind external
test. The audit IDs and labels must stay out of normal training,
validation, and model-selection paths; only evaluate them after configuration
freeze. Report scenario counts after the split without resampling or dropping
hard examples.

## Decision task

Native task: given audio and video, localize the sounding entity with spatial
metrics. A bounded Decision adaptation is currently being screened: take a
2-second audio crop centered on a Single-Sound annotation timestamp and choose
which of four timestamped video frames contains the matching annotated sound
category. Options are opaque IDs; category names, source identity, and targets
are excluded from model input. This is a cross-clip audio-to-video category
match, not same-scene entity localization and not the native AVATAR task. The
separate native multi-entity and off-screen scenarios are not represented in
this first option set; mixed-sound rows remain excluded because they do not
have a single winner.

This adaptation tests narrow audio-video category matching, not generic AV
understanding. Since audio-only or video-only recognition may solve it, measure
option-only, metadata-only, audio-only, video-only, joint, missing-stream, and
within-split audio/video-shuffle controls before claiming useful fusion.

### 2026-11-02 corrected cross-source follow-up

The corrected category-match adaptation now has a fixed-control development
result in [`R2_AVATAR_CROSS_SOURCE_READOUT_20261102.md`](R2_AVATAR_CROSS_SOURCE_READOUT_20261102.md).
On 1,655 validation choices from 305 audio-query source groups, joint accuracy
was 89.79% versus 31.18% video-only; removing audio, removing video, or
shuffling audio reduced accuracy to 23.81%, 25.02%, and 24.23%. A paired
2,000-replicate source-group bootstrap for joint-minus-video accuracy was
+55.14 to +62.49 percentage points. These controls are consistent with audio
adding information beyond the video-only arm in this constructed development
task. They do not establish the formal both-input-required gate because a
deployable matched audio-only baseline is unavailable (the category-oracle
arm is not deployable). Validation was used for checkpoint selection, exact
VideoPrism pretraining overlap is unknown, and accuracy is below the provisional
95% target. Native AVATAR localization remains unverified; this does not close
the broader M2 task matrix or justify product promotion.

Schema inspection on train/validation found per-frame `video_id`,
`frame_number`, original dimensions, FPS, duration, and annotation arrays with
`segmentation`, `bbox`, `task`, and `audio_visual_category`. The dataset-card
example currently calls the field `scenario`, but the pinned downloaded JSONs
use `task`; the official AVATAR example parser also reads `task`. The local
archive schema and parser therefore take precedence over the card example.
Sources: [dataset card](https://huggingface.co/datasets/mipal/AVATAR),
[official example parser](https://huggingface.co/datasets/mipal/AVATAR/blob/main/code/print_example.py),
and [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Choi_Whats_Making_That_Sound_Right_Now_Video-centric_Audio-Visual_Localization_ICCV_2025_paper.pdf).
The observed JSON-record and instance-task counts are:

| Split | JSON frame records | Single-Sound | Multi-Entity | Mixed-Sound | Off-Screen | Instances |
|---|---:|---:|---:|---:|---:|---:|
| Train | 19,159 | 12,006 | 3,083 | 7,423 | 540 | 23,052 |
| Validation | 2,523 | 1,693 | 345 | 887 | 70 | 2,995 |

Counts are annotation instances, not mutually exclusive frame counts. Mixed-
sound annotations remain multi-label diagnostics, and off-screen examples
need a separately defined `none_visible` construction.

The deterministic option-construction preflight found 11,995 train and 1,693
validation examples. Each has one target and three negatives from distinct
sound categories and distinct source-video groups within that split. No eligible
row was skipped, and option positions are near-balanced. The targets span
2,268 train and 305 validation source-video groups. The model input carries a
2-second audio window centered at `frame_number / fps` and candidate-frame
timestamps calculated from each candidate's own `frame_number / fps`; category
strings remain in a separate analysis sidecar. Boundary-crossing audio windows
are zero-padded: 2,230 train and 300 validation examples. These are candidate
counts, not evidence that the audio and video streams are both necessary.
Validation has 305 independent target groups, and frozen-encoder pretraining
overlap is unresolved.

Across the train/validation choice inputs, 68,440 audio/video asset references
collapse to 13,688 unique timestamped assets and 3,009 unique source MP4 clips.
All 3,009 referenced MP4 containers opened from the pinned archive with
readable audio and video streams. All referenced annotation times were within
container duration. This checks container/stream availability and time bounds;
it does not decode every media payload or confirm every timestamp's
frame/audio alignment. Full decode/alignment was checked only on three fixed
validation examples. Receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/avatar-av-decision-20261101-attempt-02/av-media-integrity-01.json`
(SHA-256 `ab117d7b34f4de1056d124d88634a93c7bc8ef82f5994fda21137e06f83ec00a`).
No audit annotation or media was accessed.

The file-to-video identity audit normalized zero-padded clip-folder offsets
against each JSON `video_id` and found zero mismatches: train has 3,952 clips /
3,286 source YouTube groups, and validation has 520 clips / 411 groups. All
visible-source boxes have positive extent and remain within frame bounds
(22,512 train and 2,925 validation); the only missing boxes are the 540 / 70
Off-Screen instances. A potential temporal subset with at least four
Single-Sound annotation records and at least two categories contains 109 train
clips / 101 source groups and 18 validation clips / 15 groups. This is only an
upper bound on useful cross-event data: sparse annotated timestamps do not
prove sound activity throughout neighboring windows, and the small validation
group count is insufficient for a broad quality claim.

AVATAR's native task is dense audio-visual localization, reported with spatial
metrics. A four-option Decision adaptation is a separate task and must not be
presented as native AVATAR localization performance. The smallest defensible
next probe is a frozen-feature cross-clip audio-to-video option match, built
only from train/validation timestamps with opaque randomized option IDs. PyAV
decoded a target audio stream and four option frames from three validation
examples (source groups `-EFu7tP-QEk`, `0OtOrR70M1A`, `9o2GcFviRBc`). PyAV frame
seeks were within 1.5 frame periods for all 12 requested frames. Actual audio
overlap plus declared boundary padding produced 2-second windows in all three
examples. Per-example CPU decode times were 0.2092, 0.5041, and 0.2745 seconds
(mean 0.3293 seconds for five media streams); this small smoke is not an
end-to-end throughput estimate. One inspected clip had annotation FPS
29.97003, video rate 30000/1001, and matching 10-second audio/video durations.
Before fitting, verify frame seeks and audio-crop alignment across more source
groups. Compare option-only, audio-only, video-only, joint, missing-stream, and
within-split audio/video-shuffle controls; report source-group intervals. The
held-out audit remains unopened.

The pinned EfficientAT and VideoPrism smoke receipts, three-run timing
averages, and rough extraction estimate are recorded in
[`R2_AVATAR_ENCODER_RUNTIME_SMOKES_20261101.md`](R2_AVATAR_ENCODER_RUNTIME_SMOKES_20261101.md).

## Model and execution gates

Use only already audited, pinned, locally available frozen encoders and a
small task readout. Keep total resident parameters below the roadmap's
200M engineering ceiling. Do not use ImageBind or any non-commercial model.
Train/evaluate on CPU if the live GPU guard is occupied; if GPU work is needed,
use only the existing local RTX 3080 and preserve the active Orbitune process
and all earlier checkpoints. No cloud compute, model upgrades, or changes to
legacy runs.

Pretraining-source overlap for the exact frozen encoders versus these source
videos is not yet verified. Treat that as an explicit limitation and do not
call the held-out split blind. Before declaring a quality result, verify the
actual archive hashes, video-ID uniqueness, annotation schema, split
disjointness, per-scenario support, media integrity, and exact model/processor
hashes. Save predictions and per-video metrics. Primary metrics are option
Accuracy, NLL, Brier, ECE and macro-by-scenario values; report the native
localization metrics only as supplemental comparisons.

At publication time, native localization model training and feature extraction
remained gated on completion of split, annotation-schema, media-integrity, and
fixed-control preflight. The corrected cross-source category adaptation above
has its own completed fixed-control result and does not satisfy those native
task gates. No sealed-audit labels may enter normal development.

## Status update (2026-11-02)

The native-task gates above have now been exercised on a new deterministic
source-group fold made only from the original AVATAR TRAIN annotations. The
frame-level DINOv2 cache passed ID, shape, finite-value, timestamp, and
train/validation identity checks. Frozen Video-only, Audio-only, and Joint
40-cell readouts plus audio-content shuffle/zero controls were evaluated on
that development-validation partition. Joint did not improve over Video-only
and was markedly worse calibrated; retain this as a negative result. The full
procedure and metrics are in
[`R2_AVATAR_NATIVE_SPATIAL_LOCALIZATION_20261102.md`](R2_AVATAR_NATIVE_SPATIAL_LOCALIZATION_20261102.md).
Official validation and sealed audit remain unopened, and no native-localization
quality claim is blind.
