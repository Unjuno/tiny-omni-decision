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

## Metadata-only parser prototype (2026-10-26, additive follow-up)

A local prototype read one first-record SequenceExample from the first train
shard and first validation shard using byte-range requests only. It traversed
20 context feature-map entries and extracted the schema keys, including
`episode_metadata/file_path` and `episode_metadata/episode_id`. It did not
fetch the image/state/action feature payloads or a complete record, and the
source path was hashed in memory rather than retained.

| Split sample | First record payload | Source size / ETag | Range requests / bytes | Collection-group hash |
|---|---:|---|---:|---|
| train shard 00000 of 01024 | 3,092,018 bytes | 108,381,239 / `"675c437-605e3b1511a6d"` | 43 / 896 | `b1c3f9a9e74e0b2534cfa56f3520cc0ad5306d0d5b3a686af0a8cea88b595905` |
| validation shard 00000 of 00128 | 4,810,980 bytes | 139,629,016 / `"85291d8-605e42c3df46b"` | 43 / 876 | `4f32bbf7a7fad3caad5141078b492b73a87b67b6496cfd38d014045edf30c3c8` |

The parsed context schema contained reward, observation image keys, state,
action, language instruction/embedding, episode ID/path, and terminal/step
flags. The machine-readable receipt is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\bridgedata-v2-metadata-preflight-20261026-attempt-02\metadata-schema-receipt.json`,
SHA-256 `B874FA9F3A36BC47DEE420215B0B9EF026E1E2A3BB62121604160BDFB301B7A3`.
The prototype is at the same artifact directory as
`range_metadata_prototype.py`.

This is only a parser feasibility check on two records. It does not establish
record counts, collection-group distributions, source split independence, or
resolve the discrepancy between the Berkeley directory metadata and the TFDS
catalog. BridgeData remains REVIEW; no media payload, training example, or
model input was acquired.

## Correction: the compared counts are from different exports

A pinned read of the TensorFlow Datasets `Bridge` importer source at commit
`aa6bbf949e676904cb2ce4c34e46113d7cd43021` shows that the TFDS catalog entry
`bridge` resolves to the relative dataset location
`bridge_data_v2/0.0.1` (see the
[pinned importer source](https://github.com/tensorflow/datasets/blob/aa6bbf949e676904cb2ce4c34e46113d7cd43021/tensorflow_datasets/robotics/rtx/rtx.py)).
The Berkeley directory inspected above is instead named
`bridge_dataset/1.0.0`; its `dataset_info.json` declares module
`bridge_dataset.bridge_dataset_dataset_builder`, and its shard lengths sum to
53,192 train plus 6,872 validation records. The TFDS catalog's 25,460 train
plus 3,475 test counts describe its separate `bridge` / `bridge_data_v2`
export, not the same identified prepared directory.

Therefore the earlier wording that characterized these as an unresolved count
discrepancy for one dataset was too strong. The count comparison is not
apples-to-apples and should not be used to infer missing or duplicated data.
The specific identity/count ambiguity is resolved by the distinct builder
name, version, storage location, and shard metadata. This does **not** establish
that the two exports have disjoint underlying episodes or source groups. The
candidate remains REVIEW until selected records are inspected metadata-only,
grouped by collection identity, and split integrity is established. No payload
or full shard was downloaded for this correction.
