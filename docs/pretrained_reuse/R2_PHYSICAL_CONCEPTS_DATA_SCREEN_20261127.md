# Physical Concepts source and data screen (2026-11-27)

## Decision

Admit the official `freeform/train` component for a bounded Path A training
pilot after input-specific frozen-feature runtime checks. Do not treat this
source screen as a model-quality result. Two downloaded shards have 6,145
records; the pinned publisher documentation describes 100 training shards and
300,000 videos in the complete freeform training split.

This is a stronger next data candidate for a temporal physical-reasoning
experiment than repeating another IntPhys dev analysis. It provides 15-frame
videos, per-frame segmentation masks and camera pose, plus separate
possible/impossible probe families. It does not yet prove that a compact
frozen representation can learn the target task.

## Provenance and license

- Publisher: Google DeepMind.
- Repository revision: `baa2776017ea3cc8c873ce682bb6646860e12cae`.
- The pinned repository says the dataset materials are CC BY 4.0; its software
  is Apache-2.0. Attribute the dataset and retain the source notices in any
  derived redistribution.
- Data are hosted in the public `gs://physical_concepts/` bucket. The shard was
  retrieved directly without credentials or personal information.
- Pinned object: `freeform/train/data.tfrecord-00000-of-00100`, generation
  `1600878343269200`, 171,007,593 compressed bytes, GCS MD5
  `LRni2RcBPHxV7wjo7fToRA==`.
- Observed SHA-256:
  `49d17f80766659c1d6e5ca760c34609e6f09f4746045740a0e669a39c123826b`.
- Second pinned object: `freeform/train/data.tfrecord-00001-of-00100`,
  generation `1600878343249041`, 174,870,282 compressed bytes, GCS MD5
  `XPwige0GtlYz7vDseopMBw==`, SHA-256
  `97659a1c78e922aef880f84c123856b39ab98261fcace7f2ae052bee0e6f2ade`.
- Local payload is outside Git at
  `C:/CodexArtifacts/pretrained-reuse-r2-local/physical-concepts-patha-20261127-attempt-01/data.tfrecord-00000-of-00100.gz`.

The exact pin, object receipt, observed counts, and audit output location are
also recorded in
[`physical-concepts-freeform-shard-00000-20261127.json`](../../manifests/pretrained_reuse/physical-concepts-freeform-shard-00000-20261127.json).
The provisional whole-shard split and zero-overlap audit are in
[`physical-concepts-freeform-two-shard-split-20261127.json`](../../manifests/pretrained_reuse/physical-concepts-freeform-two-shard-split-20261127.json).
The label-blind similarity report summary is separately pinned in
[`physical-concepts-freeform-two-shard-similarity-20261127.json`](../../manifests/pretrained_reuse/physical-concepts-freeform-two-shard-similarity-20261127.json).

## Actual data audit

The standard-library auditor checked the compressed object size and GCS MD5,
all TFRecord framing CRC32C values, protobuf feature schema, and all decoded
row shapes. The two shards contain:

| Measure | Shard 00000 (train) | Shard 00001 (validation) | Combined |
| --- | ---: | ---: | ---: |
| Records | 3,046 | 3,099 | 6,145 |
| Unique serialized rows | 3,046 | 3,099 | 6,145 |
| Duplicate serialized rows | 0 | 0 | 0 across shards |
| Unique complete RGB sequences | 3,046 | 3,099 | 6,145 |
| Duplicate complete RGB sequences | 0 | 0 | 0 across shards |
| Unique complete mask sequences | 3,046 | 3,099 | 6,145 |
| Frames per sequence | 15 | 15 | 15 |
| RGB shape | 64 x 64 x 3 uint8 | 64 x 64 x 3 uint8 | 64 x 64 x 3 uint8 |
| Mask shape | 64 x 64 uint8 per frame | same | same |
| Camera pose | 6 finite float32 values per frame | same | same |

The full record-level SHA-256 and media-sequence hashes are in the local audit
JSON outside Git. They are deliberately not copied into the repository.

These checks establish parseability, pinned provenance, and no exact
within- or cross-shard duplicate sequences. The provisional split assigns
shard 00000 to train and shard 00001 to validation; the tested cross-shard
auditor records zero exact row/RGB/mask overlaps. The source records do not
expose explicit scene or procedural-generation IDs, so related or near-duplicate
scenes may still cross the split. The label-blind appearance screen found many
nearby coarse signatures, but its group breakdown suggests repeated generator
background/camera structure rather than verified duplicate videos. Use this
split only for in-distribution freeform validation; do not present it as a
scene-OOD split.

## Label-blind scene-similarity screen

The script `scripts/audit_physical_concepts_scene_similarity.py` revalidated
both pinned files and row hashes, then formed a 524D descriptor per video from
8x8 RGB block mean/std over time, 8x8 mask occupancy mean/std, and camera-pose
mean/std. Each group and final vector was L2-normalized. Cosine similarity was
computed across train and validation; no examples were removed or reassigned.

| Statistic | Train to validation nearest | Validation to train nearest |
| --- | ---: | ---: |
| Median cosine | 0.9756 | 0.9751 |
| 90th percentile | 0.9883 | 0.9884 |
| 95th percentile | 0.9906 | 0.9906 |
| 99th percentile | 0.9939 | 0.9935 |
| Maximum | 0.9977 | 0.9977 |
| Records with nearest cosine >= 0.99 | 186 | 186 |
| Records with nearest cosine >= 0.995 | 15 | 14 |

The feature groups explain the combined score:

| Descriptor group | Train -> validation nearest p50 | Count >= 0.99 | Count >= 0.995 |
| --- | ---: | ---: | ---: |
| Coarse RGB appearance | 0.9961 | 2,596 / 3,046 | 1,871 / 3,046 |
| Mask occupancy | 0.9598 | 264 / 3,046 | 85 / 3,046 |
| Camera pose | 0.99999 | 3,046 / 3,046 | 3,046 / 3,046 |

RGB and camera-pose descriptors are nearly constant across the split, so they
are weak identity signals here; the top pairs visually share the procedural
checkerboard setting while object/color details differ. Mask occupancy is more
discriminative. These signatures do not justify dropping examples or making a
new group split. They show the source has a shared visual domain and that the
current holdout measures within-domain transfer. Exact sequence hashes still
show no clip-level duplication.

The top eight train/validation pairs were visually reviewed using first,
middle, and last frames. They share procedural checkerboard-like layouts, but
colors and object details vary. This does not prove duplicate scene identity;
the descriptor is a coarse heuristic and the source exposes no scene ID.
Preserve the split for within-domain freeform prediction-loss selection, but do
not treat it as a scene-family OOD test. The initial v1 screen output was
invalid because it omitted final-vector normalization; it is retained outside
Git and superseded by corrected v2, then the feature-group breakdown in v3.

Corrected v2 report SHA-256:
`785d4ba783c8c61a603ab6a495be6857d1fe12d61262176a73ca01dd9f359ac7`.
Feature-group breakdown v3 SHA-256:
`ca8c6051be94be4cbbb5924674ad5436329c6cce0e4b50d38788e482a1521055`.
Contact sheet SHA-256:
`085c3d3ebba250c299fbd853c5ee997b3e908abe2dd52ca0ad51495ec542823a`.

## Task and split plan

The useful task is temporal physical consistency. A proposed Path A experiment
would fit a small predictor over pinned frozen VideoPrism features computed
from the freeform training split. Before opening probe sets, reserve whole
freeform shards for validation and fix temporal sampling, predictor settings,
and scoring using only that split. Then perform a single task-specific probe
evaluation on the official possible/impossible probe families.

The publisher describes five probe families (Continuity, Solidity, Object
Persistence, Unchangeableness, and Directional Inertia) with related video
quadruplets. Probe sets are evaluation-only in this plan. None of their files or
labels were downloaded or opened. This preserves a clean next evaluation and
avoids using a public probe repeatedly for tuning.

## Relation to the current evidence and BERT

The IntPhys Path A result was near chance on its fixed development diagnostic.
That is evidence against that specific possible-only anomaly-transfer recipe,
not evidence that all temporal models or all video data are unlearnable. The
IntPhys official training protocol is unsupervised and contains possible-only
examples, so its label/task mismatch is a substantive constraint.

BERT-family work is still relevant for text-input Decision tasks. It is not a
substitute for a video temporal representation and should not be added to this
video experiment. Since prior BERT results are task-specific and some
evaluation splits are already exposed, a new BERT comparison should wait for a
newly admitted, source-disjoint text Decision dataset with explicit correctness
labels. For the current temporal question, improving the data/task match is
the higher-priority move than increasing text-model capacity.

## Temporal signal and pinned encoder input smoke

A label-free screen covered all 6,145 rows. Every video has measurable RGB
motion, a mask change, and camera-pose change; each has 15 distinct RGB frames,
and no adjacent RGB transition is byte-identical. Median per-video adjacent RGB
MAE is 0.0221 (train) / 0.0223 (validation), median changed-mask pixel fraction
is 0.0168 / 0.0168, and mean camera-pose delta L2 is 0.5050 / 0.5052. These
statistics are similar across the two shards. They establish temporal signal in
the records, not predictability or physical-reasoning quality. The result is
saved outside Git in
`C:/CodexArtifacts/pretrained-reuse-r2-local/physical-concepts-patha-20261127-attempt-02/temporal-signal-v1.json`
(SHA-256 `6110acbb122e6f42a0dbd7218da7328f482b374aa4203de9baeea1fa3b456d85`).

The R1-pinned VideoPrism-B checkpoint and CPU environment were run on 16 actual
train videos. The first compiled forward took 6.78 seconds; the next 15 took
1.14–1.23 seconds each (mean including compilation: 1.53 seconds). Output
shape is `[16,8,768]`, with temporal order retained. Peak working set was
4,136,878,984 bytes (3.85 GiB); available RAM was 4.43 GiB before load,
4.02 GiB after load, and 3.07 GiB after forward. The measured steady-state rate projects to
about 2.1 hours for both current shards, before decompression and archive I/O.
This is runtime evidence only. Feature output and receipt are outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2-local/physical-concepts-videoprism-smoke-20261127-attempt-02/`;
the receipt SHA-256 is
`d53718630b57cc265976d3e889ac5691b15f12b8673fee130e491f5a08898332`.

Full extraction for train and validation is a separate CPU-only run in a new
attempt directory. The frozen features feed the existing small next-feature
predictor (seed 17, hidden width 64, AdamW at 1e-3, maximum 100 epochs, batch
64, patience 8, min-delta 1e-4); selection is minimum validation cosine loss.
The report will compare that selected loss with the no-training persistence
baseline on the exact same validation features. No probe data or labels have
been opened.

Train feature extraction has completed on all 3,046 rows. The cache has shape
`[3046,8,768]`, all values are finite, and all 3,046 sample and group IDs are
unique. The compressed file is 69,929,470 bytes (SHA-256
`c9445a1a5f3f236ec768200006024d88cdf8e935be1b90c0a35ec3c429c0a816`); the
feature-value SHA-256 is
`e1c947fe747064162e1653d0707e0904c15c3a6aa883cf118134096e2ae5abc2`. Actual
mean forward time was 1.2000 sec/video, or 3,655.2 seconds summed across the
train shard. The process started at 11:23:11 JST and completed by 12:26:24
JST (about 3,793 seconds wall time, including load, decode, and archive save).
Peak working set was 4,131,540,992 bytes. The saved artifacts are under
`C:/CodexArtifacts/pretrained-reuse-r2-local/physical-concepts-videoprism-features-20261127-attempt-01/train/`;
the receipt SHA-256 is
`396dcc4c5a8d0a7ac8221e2c9bb435fd44b2d3198a7720d674e01aad481d6dd0`.

The first extractor implementation wrote its `started_at_utc` field after
feature creation. That train receipt is preserved as written; elapsed time
above is derived from the Windows process creation time and that completion
timestamp. The script now records start and end times at the correct points.
Validation extraction was already running when that correction landed, so its
duration will be derived the same way from its process creation record.

## Outstanding gates

- Validation feature extraction and a validation-selected Path A predictor
  result remain pending. The separate R1-pinned environment uses
  JAX 0.6.2 CPU; no packages were upgraded and no GPU work was started.
- Keep validation use in-domain: select against the shard-held-out freeform
  prediction loss only, and report that it does not test new scene families.
- Freeze the Path A scorer and validation selection rule before any probe
  evaluation.
- Report task-level accuracy and uncertainty honestly; do not generalize a
  probe result to four-modality product quality.
