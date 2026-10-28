# GOOSE instance-label mapping correction (2026-10-28)

Status: **mapping-aware integrity recheck complete; GOOSE remains unadmitted as an R2 M-2 Decision task.** This correction does not replace or edit the original audit or its artifacts.

## Why this recheck was needed

The earlier probe compared every nonzero `(semantic_id, instance_id)` pair and discarded `instance_id == 0`. The official `goose_label_mapping.csv` distinguishes classes that are instance-labeled from semantic-only classes. The publisher's [labeling policy](https://goose-dataset.de/docs/resources/labeling_policy.pdf) also gives `car 0` as a valid instance ID and says the same ID is used across camera and LiDAR sensors. The [dataset structure](https://goose-dataset.de/docs/dataset-structure/) documents the 3D semantic/instance bit packing.

Thus the prior key counts were not computed under the full published annotation semantics. The correction filters to the 26 mapping rows with `has_instance=1` and retains zero-valued instance IDs. The mapping CSV was read from both official train archives and was byte-identical; SHA-256: `88ae319ba5a3877dd3ae0773f693a6a5fdc283934140de9dfaff029108aefd7f`.

## Frozen recheck

- Source archive-index SHA-256: `38fa35593065b5768e823dc67275d7088c359b29e1690724bfc3f54178d76cb4`.
- Label mapping SHA-256: `88ae319ba5a3877dd3ae0773f693a6a5fdc283934140de9dfaff029108aefd7f`.
- Reused the same deterministic 15 train-only scenarios as the prior combined audit; all eight scenarios represented in validation remain excluded.
- Each of the 75 selected archive members passed pinned HTTPS Range/ETag, local ZIP header, size, decompression, and CRC checks.
- The source payload was decoded in memory and discarded. No validation or test payload, model, GPU, or sealed audit was used.

## Results and comparison

| Measure | Prior combined audit | Mapping-aware recheck |
|---|---:|---:|
| Train-only scenarios | 15 | 15 |
| 2D object keys | 561 | 555 |
| 3D object keys | 1,379 | 1,379 |
| Shared `(class, instance)` keys | 283 | 283 |
| Shared / 2D keys | 50.4% | 51.0% |
| Median per-scenario shared / 2D keys (14 with image instances) | 57.3% | 59.0% |
| Per-scenario range | 9.1%–100% | 9.1%–100% |
| Scenarios at >=90% / below 50% | 2 / 7 | 3 / 7 |

The class-mask and zero-ID rule changed the 2D total by six keys net, while the shared-key count stayed at 283. The observed overall and per-scenario correspondence therefore remains highly variable. The aggregate does **not** establish that the source is intrinsically bad: visibility, temporal alignment, occlusion, and the one-frame-per-scenario sampling design remain plausible causes. In particular, label-key overlap is only a supervision-integrity screen; it is not a model-quality metric.

The correction does not provide a non-oracle inference-time object-proposal path or a reliable decision target. Do not admit GOOSE to Decision training/evaluation on this evidence. Keep R2 M-2 open and prefer a task with a defined inference-time input/target before spending further compute on this source.

## Artifacts and verification

- Corrected result JSON (external, immutable): `C:/CodexArtifacts/pretrained-reuse-r2/goose-instance-map-reconciliation-20261028-attempt-04/result.json`.
- Corrected result SHA-256: `97d24e5cf5cc7cc725c5a51b366f572dd81cd1691a824e20a2207bf6896ad8de`.
- 15-frame transfer: 87,337,805 bytes in 478.6031 summed Range seconds; pooled 0.174 MiB/s. Raw payloads were not retained.
- The original report and artifacts remain unchanged. This recheck adds mapping-aware unit tests and fails closed on an unexpected mapping hash.
