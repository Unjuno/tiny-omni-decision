# BridgeData V2 TFDS metadata follow-up (2026-10-26)

## Finding

The official Berkeley TFDS directory at
`https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/`
contains 1,024 train shards and 128 validation shards. Its pinned-in-place
`dataset_info.json` reports 53,192 train and 6,872 validation episodes, with
`numBytes` values of 117,084,911,456 and 15,401,054,101 respectively. This
differs materially from the TensorFlow Datasets catalog, which reports 25,460
train and 3,475 test episodes and a 387.49 GiB prepared dataset. The catalog
and Berkeley directory appear to describe different prepared views or
revisions; the difference is unresolved. Do not treat the counts or split names
as interchangeable.

The Berkeley listing shows individual TFRecord shards around 90–132 MB. The
local C: drive had 5,904,158,720 bytes free during this check, far below the
aggregate `dataset_info.json` byte total. Full acquisition remains out of
scope. The BridgeData project page's CC BY 4.0 statement still makes the
source worth investigating, but rights do not resolve the group-split or
version mismatch.

## Read-only method and receipts

At 2026-10-26 04:23:59 UTC, HTTP 200 responses were saved outside Git:

- `C:\CodexArtifacts\pretrained-reuse-r2\bridgedata-v2-metadata-preflight-20261026-attempt-01\dataset_info.json`
  SHA-256 `977f2efb2a4be1d839b0b67b9cae8f2007568a1da689ad170f1b72846dc54653`
- `...\directory-index.html`
  SHA-256 `975c80df400b8d6b2c1d0748f566f8ea189f695b8abce51bc5e81c574a04077a`
- `...\receipt.json`
  SHA-256 `a1ed428ff8c3c82ad9bf4715fab48bb2b5702374cb856a0ad97ecd6d4102e612`

Two single-byte-range requests fetched only the first 32 bytes of one train
shard and one validation shard. Both returned HTTP 206 with matching
`Content-Range` and ETag. These prefixes confirm TFRecord framing and show the
first feature key starts with `steps/reward`; they do not establish the
episode-path/group distribution. No complete shard, image, sensor array, label,
or annotation was downloaded or opened. A multi-range request stalled and was
cancelled; no result or data artifact was produced by that attempt.

## Disposition and next gate

Keep BridgeData at **REVIEW / metadata-only**. Before any media acquisition or
model fitting, reconcile the prepared-view/count discrepancy and demonstrate a
metadata-only method to extract `episode_metadata/file_path` and
`episode_metadata/episode_id` across candidate records without fetching image
payloads. Then establish group-disjoint splits, measure the selected-group
download size, and predeclare an action-choice task with matched vision-only,
state-only, and paired controls. The source's raw train/validation split is
not assumed to be collection-group independent. No shard was acquired and no
model was trained in this follow-up.

## Sources

- [BridgeData V2 project page](https://rail-berkeley.github.io/bridgedata/)
- [Berkeley TFDS data directory](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/)
- [TensorFlow Datasets Bridge catalog](https://www.tensorflow.org/datasets/catalog/bridge)
- [TFDS slicing and determinism documentation](https://www.tensorflow.org/datasets/splits)
