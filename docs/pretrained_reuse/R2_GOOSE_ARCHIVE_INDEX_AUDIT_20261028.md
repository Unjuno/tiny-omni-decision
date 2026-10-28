# GOOSE archive index, scenario split, and transfer preflight (2026-10-28)

Status: **PASS for bounded source-index integrity and split diagnosis; official
split is not scenario-disjoint. A filtered 15-train / 8-validation scenario
cohort is feasible, but no both-input decision task or quality result is
established.** No model was trained or evaluated.

## Scope and safeguards

The official publisher declares the GOOSE dataset CC BY-SA 4.0 and identifies
its archive contents. Its dataset structure defines a `scenario` as one
measurement run and a `sequence` as one ROS bag within a scenario. The archive
directory names therefore provide a source-group key suitable for detecting
scenario overlap. See the publisher's [license and downloads](https://goose-dataset.de/docs/setup/),
[scenario/sequence definition](https://goose-dataset.de/docs/dataset-structure/),
and [project overview](https://goose-dataset.de/).

Only the EOCD/ZIP64 records and central-directory ranges of the four official
train/validation archives were fetched. The audit did not read image, point
cloud, or label payloads. It separately fetched three approximately 2.5 MB
RGB train members to measure transfer speed, verified each local header and
ZIP CRC, computed SHA-256 values in memory, and discarded the decoded bytes.
It did not save or display those images. No validation payload, annotation,
test archive, model, sealed audit, GPU, cloud compute, or user checkpoint was
accessed. The official test split remains untouched.

The archive's full SHA-256 is **UNKNOWN**: the publisher exposes no archive
checksum on the inspected pages, and this bounded preflight did not download
whole archives. Each archive was pinned for this audit by its HTTPS URL,
`Content-Range` size, ETag, Last-Modified header, and fetched central-directory
SHA-256. This pins the observed index, not the integrity of every media byte.
Any later use must verify each selected member against its central-directory
CRC and saved member SHA-256.

## Exact archive metadata

| Archive | Bytes | ETag | Last-Modified (UTC) | Entries | Scenario groups | Asset count | Central-directory SHA-256 |
|---|---:|---|---|---:|---:|---:|---|
| `goose_2d_train.zip` | 22,466,451,989 | `"67d9cad6-53b1ada15"` | 2025-03-18 19:34:46 | 39,228 | 23 | 7,845 RGB images | `00e4cb39594689b56f7bc947462900f94bd76abe6672638e5a0d54e389769614` |
| `goose_2d_val.zip` | 2,857,251,081 | `"67d9a658-aa4e3109"` | 2025-03-18 16:59:04 | 4,813 | 8 | 962 RGB images | `3d23a0b213154b37d4ca934ddc0d58fd300d7669e5891dbf18b06921a5833760` |
| `goose_3d_train.zip` | 28,846,555,347 | `"67daa5b4-6b76380d3"` | 2025-03-19 11:08:36 | 15,441 | 23 | 7,719 point clouds | `06776988a3ab5fff529814f2c577dd9c61bb444b844655e8b6360b7e081f8c12` |
| `goose_3d_val.zip` | 3,498,402,435 | `"67da9ea9-d0856283"` | 2025-03-19 10:38:33 | 1,925 | 8 | 961 point clouds | `bf06e8e1b8360641daafd5c8479f2bde286bea1d9e9a8ec495b0d2b00eb32873` |

The publisher's page lists approximate 3D validation size as 3.3 GB; the
observed range response reports 3,498,402,435 bytes (3.50 decimal GB). The
observed 3D train size is 28.85 GB versus the page's approximate 27 GB. Use the
range-pinned values for storage planning until the publisher reconciles these
figures.

## Split and cross-modal findings

| Split policy | Train groups | Validation groups | Exact train/validation asset overlap | Matched image/point-cloud IDs |
|---|---:|---:|---:|---:|
| Publisher split, image | 23 | 8 | 0 | — |
| Publisher split, point cloud | 23 | 8 | 0 | — |
| Publisher split, train and validation share scenario groups | **8** | **8** | 0 exact frame IDs | — |
| Filtered scenario-disjoint train | 15 | — | 0 by group construction | 4,473 pairs |
| Validation scenario cohort | — | 8 | 0 exact frame IDs | 961 pairs |

The same eight scenario directory IDs occur in both publisher train and
validation archives for both modalities. The exact image/point-cloud frame IDs
do not repeat across the two splits, but a same-scenario split does not meet
the repository's scene/source-disjoint requirement. Do not use the publisher's
unfiltered train/validation split for quality claims.

As a deterministic source-split repair, removing every train member from the
eight scenarios present in validation leaves 15 train-only scenarios. This
gives 4,604 train images, 4,476 train point clouds, and 4,473 exact paired
assets, plus 962 validation images, 961 point clouds, and 961 exact pairs from
eight disjoint scenarios. These are index counts, not independent examples or
a completed usable corpus. The original validation scenario cohort has
already been examined at the metadata level and must be treated as development
data; it is not a blind audit.

The cross-modal filenames join exactly for 7,712 of 7,845 publisher-train
images and 961 of 962 publisher-validation images. Seven train point clouds
and one validation image lack an exact counterpart under the publisher's
filename convention. No attempt was made to infer alternate joins.

## Three-sample transfer measurement

For a bounded speed estimate, the audit selected three train RGB members from
three distinct scenario groups, each closest to 2,500,000 compressed bytes.
Each member was fetched through two exact HTTP range requests (local header,
then filename/extra/payload), checked against the pinned ETag and central
directory, CRC-verified after in-memory decode, and discarded.

| Sample | Bytes including local header | Seconds | MiB/s | CRC |
|---|---:|---:|---:|---|
| 1 | 2,500,323 | 10.202 | 0.234 | pass |
| 2 | 2,500,393 | 6.050 | 0.394 | pass |
| 3 | 2,499,834 | 7.510 | 0.317 | pass |
| Mean per sample | — | — | **0.315** | — |
| Pooled | 7,500,550 | 23.762 | **0.301** | — |

At the pooled rate, both complete validation archives (6,355,653,516 bytes)
would take about **5.59 hours**. This is an extrapolation from three
sequential train-image range reads, not a promise: point-cloud throughput,
long-running connection stability, host load, and future network conditions
may differ. The index-only pass transferred 9,955,579 bytes across 12 ranges
at 0.213 MiB/s effective, which is not used for the full-data estimate.

## Reproduction and external artifacts

Commands:

```powershell
python -m scripts.audit_goose_archive_indexes `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\goose-index-audit-20261028-attempt-03

python -m scripts.probe_goose_train_transfer `
  --index-path C:\CodexArtifacts\pretrained-reuse-r2\goose-index-audit-20261028-attempt-03\archive-indexes.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\goose-transfer-probe-20261028-attempt-01
```

- Index inventory SHA-256: `38fa35593065b5768e823dc67275d7088c359b29e1690724bfc3f54178d76cb4`.
- Index result JSON SHA-256: `108d7ad3a600746df28119ca39d86998bd48cbb94614e34cf10a8e2e349d02a5`.
- Three-sample plan SHA-256: `5099a7bd917c92dacb4845936e8bf07249cfbd33cf3d25f580323da663b675f4`.
- Three-sample result JSON SHA-256: `418b14a97cab38d962be1b88ab95ba08443ae815c94e2c878578d3f05e6dda2f`.
- Index audit script SHA-256: `cf418c4dc4a98cbdb1bd6a523417196a590e2386b58a30a21bb3d15712cf57ea`.
- Transfer probe script SHA-256: `03219c2c4cee80f446a98a454535b61a752191a7db6ad6584a44e7d727fd3c03`.
- Focused test file SHA-256: `740fca7141f42b4acb5322d89a19f795905fd3cd89174aa347200a770e3d374a`.
- External attempt directories: `C:/CodexArtifacts/pretrained-reuse-r2/goose-index-audit-20261028-attempt-03/` and `C:/CodexArtifacts/pretrained-reuse-r2/goose-transfer-probe-20261028-attempt-01/`.

The focused tests pass (6); Ruff and `py_compile` pass. The host had 10.96 GiB
free on C: after this bounded transfer. No archive or media file was added to
Git. Existing Teacher, v1/v2, checkpoint, sealed-audit, and Orbitune run files
were not modified; the occupied RTX 3080 was not used.

## Interpretation and next gate

GOOSE now supports an executable, scenario-disjoint camera/point-cloud source
cohort and has a reproducible transfer estimate. It still does **not** prove a
Decision target that requires both image appearance and 3D geometry: the
released annotations are semantic segmentation labels. Before extracting a
corpus or fitting readouts, define a task whose correctness genuinely depends
on both inputs, then freeze group-safe train/validation IDs and controls. If no
such task can be justified without manufacturing labels from the input
features, retain GOOSE as a paired-modality engineering probe only; do not call
it the R2 M-2 quality gate or promote it to a product task.
