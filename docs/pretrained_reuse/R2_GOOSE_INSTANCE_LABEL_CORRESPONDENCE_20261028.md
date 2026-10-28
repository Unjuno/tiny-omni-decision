# GOOSE cross-sensor instance-label correspondence probe (2026-10-28)

Status: **bounded data-integrity probe passed; a camera/LiDAR object-match task
is not admitted. The first three fixed train-only frames show inconsistent
class-and-instance correspondence, requiring explanation before corpus use.**
No encoder or Decision model was loaded, trained, or evaluated.

## Protocol

The probe used the pinned GOOSE train/validation ZIP-index inventory
(`archive-indexes.json`, SHA-256
`38fa35593065b5768e823dc67275d7088c359b29e1690724bfc3f54178d76cb4`). It
excluded all eight scenario groups observed in validation and selected the
lexicographically first complete paired frame from each of the first three
remaining train-only groups. Each frame required the RGB image, 2D semantic and
instance masks, XYZI point cloud and SemanticKITTI-format 3D labels. Raw
scenario/frame names are represented only by hashes in the external artifacts.

The publisher's [dataset structure page](https://goose-dataset.de/docs/dataset-structure/)
specifies image `labelids` and `instanceids`, one `vls128.bin` LiDAR scan per
frame, and `.label` semantic IDs in the low 16 bits and instance IDs in the high
16 bits. The publisher's [labeling policy](https://goose-dataset.de/docs/resources/labeling_policy.pdf)
states that an object's instance ID should persist across consecutive frames
and across sensors. The train RGB and 3D archives contain byte-identical
`goose_label_mapping.csv` files (SHA-256
`88ae319ba5a3877dd3ae0773f693a6a5fdc283934140de9dfaff029108aefd7f`), so the
semantic IDs used in this check have one observed mapping. The source is
published CC BY-SA 4.0; any later corpus or derived artifact must retain the
required attribution and licensing review.

For each of three fixed frames the script fetched five ZIP members by bounded
HTTP byte ranges, pinned to the source ETags and central-directory records.
All 15 members passed local-header, size and ZIP CRC checks. PNGs and point
arrays were decoded only in memory; only hashed member identities, payload
SHA-256 values and aggregate label counts were retained. The run did not touch
validation or test payloads, model weights, GPU, sealed audit or existing
checkpoints.

## Observed labels

| Scenario hash | 2D instance IDs | 3D instance IDs | Numeric ID overlap | Same `(semantic class, instance ID)` | 2D instance coverage |
|---|---:|---:|---:|---:|---:|
| `b0e114224391d1a2` | 25 | 55 | 25 | 25 | 100.0% |
| `f4c662f4ab934f05` | 17 | 166 | 17 | 7 | 41.2% |
| `a9b1a3d8ce0f85cc` | 16 | 75 | 16 | 14 | 87.5% |

"Same class and instance" counts compare the actual class/instance tuples from
both label formats; a numeric ID collision alone is not treated as an object
match. The three images decoded as 2048×1000 RGB. Their point clouds contained
205,130–214,424 finite XYZI points. The 3D scan can contain objects outside the
forward camera view, so 3D-only IDs are expected and the Jaccard of all 2D/3D
instances is not a sensor-consistency metric. More concerning, 10 of 17
numerically overlapping IDs in the second frame and 2 of 16 in the third had a
different modal semantic class across modalities, despite the matching CSV.
These disagreements could reflect timing/view/annotation conventions or a
source inconsistency; three frames cannot distinguish those explanations.

The official policy makes a cross-sensor identity task plausible, but this
probe does not verify that the released ready-to-use labels implement that
contract for arbitrary frame pairs. Do not make class/instance matches into
Decision labels yet. First establish the correct visibility and object-key
semantics on a broader, preregistered train-only sample and identify an
inference-time proposal path that does not expose ground-truth masks to the
Decision model. No quality metric or Path A result is claimed here.

## Transfer and artifacts

- Run: `C:/CodexArtifacts/pretrained-reuse-r2/goose-instance-correspondence-20261028-attempt-03/`.
- Plan SHA-256: `9cb8dbdb7c4955953dd94e1c17c34a276c4b25bbec83b30808c01f6ca1be30b4`.
- Result SHA-256: `3789bc2234c75173434fabe769350124f95dcc2548931624f223e7c655ee71df`.
- Source script: `scripts/probe_goose_instance_correspondence.py`.
- Focused suite: eight tests pass, including train/validation scenario exclusion,
  cross-modal instance-key matching, ZIP64/index checks and CRC verification.
- Transfer: 19,801,052 bytes / 83.7622 summed range seconds; member mean
  0.167 MiB/s, pooled 0.225 MiB/s. The estimate is only for these 15 members.
- Three earlier RGB-only speed samples averaged 0.315 MiB/s per sample; their
  validation-archive estimate remains a rough 5.59 hours, not a download
  commitment. No complete archive was downloaded.

## Gate decision

The bounded index and selected-member integrity gates pass. The data show a
candidate cross-sensor target but not yet reliable object correspondence, and
the point-cloud pretrained checkpoint/runtime gate is still open. Keep GOOSE
as a source candidate. No model-selection, training, sealed-audit, or product
claim is justified by this probe.
