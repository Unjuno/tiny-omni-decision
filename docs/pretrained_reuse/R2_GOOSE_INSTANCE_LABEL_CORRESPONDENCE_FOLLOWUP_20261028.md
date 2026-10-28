# GOOSE train-only instance-label correspondence follow-up (2026-10-28)

Status: **15-group label-integrity screen complete; GOOSE cross-sensor
instance matching is not admitted as an R2 M-2 Decision task.** The check found
highly variable image-to-point-cloud `(semantic class, instance ID)` agreement.
No pretrained encoder, Decision head, GPU, or validation/test payload was
used.

## Frozen sample and method

The exact ZIP-index inventory was pinned by SHA-256
`38fa35593065b5768e823dc67275d7088c359b29e1690724bfc3f54178d76cb4` and the
archive URL, size, ETag, Last-Modified and central-directory hashes recorded in
the [initial audit](R2_GOOSE_ARCHIVE_INDEX_AUDIT_20261028.md). All eight
scenarios represented in official validation were excluded from train. A
deterministic lexical ordering of the remaining 15 train-only scenarios chose
one complete paired frame per scenario, requiring RGB, 2D semantic labels, 2D
instance labels, XYZI points and SemanticKITTI-format 3D labels. The first
three scenarios are from attempt 03; the other twelve are the disjoint
continuation in attempt 04. Group hashes show 15 unique scenarios and no
validation group.

For each frame, the probe compared sets of `(semantic class ID, instance ID)`
keys across the image and point-cloud annotations. A raw numeric instance-ID
collision without the same semantic class is not counted as the same object.
The common `goose_label_mapping.csv` in both train archives was byte-identical
(SHA-256 `88ae319ba5a3877dd3ae0773f693a6a5fdc283934140de9dfaff029108aefd7f`).
Each of the 15 frames' five archive members passed bounded HTTP range, pinned
ETag, local ZIP header, uncompressed size and CRC checks; image/label payloads
were decoded in memory and discarded. The publisher's [dataset structure](https://goose-dataset.de/docs/dataset-structure/)
defines the image and point label formats; its [labeling policy](https://goose-dataset.de/docs/resources/labeling_policy.pdf)
states that instance identity is intended to persist across sensors.

## Results

| Measure | Result |
|---|---:|
| Train-only scenarios sampled | 15 / 15 |
| Frames with at least one 2D instance key | 14 / 15 |
| 2D class-instance keys | 561 |
| 3D class-instance keys | 1,379 |
| Shared class-instance keys | 283 |
| Weighted fraction of 2D keys present in 3D | 50.4% |
| Median per-scenario 2D-key coverage | 57.3% |
| Scenario coverage range | 9.1%–100% |
| Scenarios at >=90% / below 50% coverage | 2 / 7 of 14 |
| Numeric ID overlaps, with semantic-class disagreement | 497, of which 214 disagreed |

The point cloud contains substantially more annotated object keys than the
forward RGB image, so 3D-only objects are expected and the set Jaccard is not
interpreted as an accuracy score. The concern is the variation in whether an
image object has a corresponding 3D key with both its ID and semantic class
preserved. The official cross-sensor identity intent does not match most of
these first-annotated frame pairs consistently enough for us to derive
supervision without resolving visibility, timing and label-instance semantics.

This is a deterministic one-frame-per-scenario integrity screen, not a random
population estimate or a model-quality evaluation. It does not prove that the
dataset is intrinsically unusable; it does show that the candidate target is
not safe to train/evaluate against as currently constructed. The earlier
3-frame-only report remains a separate historical attempt and is not changed.

## Transfer and artifacts

- Initial three-group result: `C:/CodexArtifacts/pretrained-reuse-r2/goose-instance-correspondence-20261028-attempt-03/result.json`, SHA-256 `3789bc2234c75173434fabe769350124f95dcc2548931624f223e7c655ee71df`.
- Twelve-group continuation: `C:/CodexArtifacts/pretrained-reuse-r2/goose-instance-correspondence-20261028-attempt-04/result.json`, SHA-256 `f11021891b08e139a2607ec3a6bfcdebb709474176fcc866e842adb8fe918e595`.
- Combined derived analysis: `C:/CodexArtifacts/pretrained-reuse-r2/goose-instance-correspondence-20261028-attempt-04/analysis.json`, SHA-256 `e69e9866b783d980e2f1836790c76ad3f3cbdeecac2730ca5600c99b80fd6045`.
- The combined transfer measured 87,337,805 bytes in 431.2417 summed member-range seconds (0.193 MiB/s pooled). Extrapolating to both official validation archives would be about 8.72 hours; this is a rough estimate from 15 train frames only, not a planned download.
- No media or raw labels were retained; only IDs represented by hashes, CRC status, payload hashes and aggregate counts were saved. The official test split and sealed audit remain untouched.
- The bounded suite has nine tests, including scenario-offset selection, label-key aggregation, zero-instance frames, ZIP64 parsing, pinned ranges and CRC validation.

## Gate decision

Do not use GOOSE class-instance correspondence as Decision training or
validation supervision until a new audit explains and resolves the group-level
variation and establishes a non-oracle inference-time object-proposal path.
Keep the source as a documented candidate, but move R2 M-2 effort to another
rights-cleared paired task rather than spending several hours downloading full
validation archives whose current labels do not support a dependable target.
This leaves R2 M-2, the broader Path A task battery, and later R3–R8 gates open.
