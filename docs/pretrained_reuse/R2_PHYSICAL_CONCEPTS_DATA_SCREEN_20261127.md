# Physical Concepts source and data screen (2026-11-27)

## Decision

Admit the official `freeform/train` component for a bounded Path A training
pilot after the frozen-feature runtime is available. Do not treat this source
screen as a model-quality result. The one downloaded shard has 3,046 records;
the pinned publisher documentation describes 100 training shards and 300,000
videos in the complete freeform training split.

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
- Local payload is outside Git at
  `C:/CodexArtifacts/pretrained-reuse-r2-local/physical-concepts-patha-20261127-attempt-01/data.tfrecord-00000-of-00100.gz`.

The exact pin, object receipt, observed counts, and audit output location are
also recorded in
[`physical-concepts-freeform-shard-00000-20261127.json`](../../manifests/pretrained_reuse/physical-concepts-freeform-shard-00000-20261127.json).

## Actual data audit

The standard-library auditor checked the compressed object size and GCS MD5,
all TFRecord framing CRC32C values, protobuf feature schema, and all decoded
row shapes. The shard contains:

| Measure | Observed |
| --- | ---: |
| Records | 3,046 |
| Unique serialized rows | 3,046 |
| Duplicate serialized rows | 0 |
| Unique complete RGB sequences | 3,046 |
| Duplicate complete RGB sequences | 0 |
| Unique complete mask sequences | 3,046 |
| Frames per sequence | 15 |
| RGB shape | 64 x 64 x 3 uint8 |
| Mask shape | 64 x 64 uint8 per frame |
| Camera pose | 6 finite float32 values per frame |

The full record-level SHA-256 and media-sequence hashes are in the local audit
JSON outside Git. They are deliberately not copied into the repository.

These checks establish parseability, pinned provenance, and no exact
within-shard duplicate sequences. They do not establish source independence,
semantic quality, sufficient temporal change, a correct Decision target, or
generalization. No train/validation split or model selection should be made by
sampling clips randomly across shards until cross-shard identities have been
audited. A future pilot must assign whole records and keep its validation
identities out of training.

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

## Outstanding gates

- An existing R1-pinned VideoPrism environment is present at
  `C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv`; it imports
  JAX 0.6.2 and selects CPU. The global/worktree environments lack these ML
  packages and were left unchanged. No forward pass on Physical Concepts data
  has been run yet. At audit time system free RAM was 5.04 GiB while the
  existing Orbitune process held about 9.2 GiB; defer model loading until there
  is enough memory headroom.
- Run a separate-process smoke on actual decoded source frames, record
  throughput and memory, and estimate the cost of feature extraction before
  deciding how many shards to use.
- Audit a source-disjoint freeform train/validation allocation and cross-shard
  exact media identities before training.
- Freeze the Path A scorer and validation selection rule before any probe
  evaluation.
- Report task-level accuracy and uncertainty honestly; do not generalize a
  probe result to four-modality product quality.
