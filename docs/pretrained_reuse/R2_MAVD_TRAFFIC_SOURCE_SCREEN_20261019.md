# MAVD-traffic source screen — 2026-10-19

## Outcome

MAVD-traffic is recorded as a **review-only candidate**, not admitted to a
training or evaluation catalog. It provides synchronized urban audio/video and
expert audio-event intervals, but its official development split is not
independent under conservative site-day grouping, and the base annotation
does not establish visual truth for those audio events. The presence of two
media streams therefore does not establish a task that requires both.

No model, corpus, training run, or sealed evaluation was changed or run for
this screen. No media was downloaded, decoded, played, or inspected. The test
annotation archive was downloaded only to pin its bytes and list ZIP member
names for grouping; no test annotation payload was opened or parsed, and no
test metrics were produced.

## Source and immutable local screen

- Upstream record: [Zenodo MAVD-traffic, record 4741232](https://zenodo.org/records/4741232), version 0.1.1.
- Dataset description: [DCASE MAVD catalog entry](https://dcase-repo.github.io/dcase_datalist/datasets/sounds/mavd.html).
- Split description: [DCASE 2019 paper](https://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Zinemanas_70.pdf).
- License metadata: CC BY 4.0 in the record metadata and dataset catalog. The source files remain subject to attribution and the project-specific media/privacy review.
- The record lists 47 recordings from four sites, synchronized audio/video, and manually annotated traffic sound events. The paper describes approximately five-minute segments; this makes file-level separation weaker than a session/day grouping.
- Record file bytes from the Zenodo API: 1,327,320,789 total; 1,327,276,703 media. These are catalog sizes, not local downloads.

The annotation/metadata-only source bundle is in
`C:/CodexArtifacts/pretrained-reuse-r2/mavd-source-screen-01`. The small
archives and metadata were checked against Zenodo MD5 values and pinned by
SHA-256 in `manifests/candidates/mavd-traffic.yaml`. The `revision` field is a
content-derived compatibility key for the local screened bundle, not an
upstream Git commit; the exact upstream record version and per-file hashes
are stored separately.

Reproduce the source audit with:

```powershell
py -3.11 -m scripts.audit_mavd_traffic_source `
  --source-dir C:\CodexArtifacts\pretrained-reuse-r2\mavd-source-screen-01 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\mavd-source-screen-01\source-audit.json
```

The script validates known file hashes, reads TRAIN and VALIDATION annotation
payloads, enumerates TEST archive member names without reading member data,
and reports interval counts and conservative site-day overlap. Unit tests
include invalid UTF-8 in a test member and verify the payload remains unread.

## Findings

| Split | Recording annotation files | Site grouping | Audio event intervals |
| --- | ---: | --- | ---: |
| TRAIN | 24 | Rivera only | 1,737 |
| VALIDATION | 7 | Rivera only | 556 |
| TEST (names only) | 16 | Acevedo 3, Magallanes 10, Templete 3 | Not read |

TRAIN and VALIDATION share three conservative site-day groups:
`rivera:2017-09-23`, `rivera:2017-10-15`, and `rivera:2017-10-16`.
They therefore cannot be treated as independent model-selection splits under
this grouping. The official TEST filenames identify other sites, but its
labels remain unopened and this observation is not an evaluation result.

The development annotations have valid start/end intervals (1,737 TRAIN and
556 VALIDATION). Their label strings are hierarchical and overlapping; the
union of observed development labels needs reconciliation with the catalog's
reported 21-class count before building a task adapter. This screen does not
change or collapse those labels.

The base dataset annotates traffic sounds. It does not independently annotate
the visible source/event for each sound interval. A defensible narrow audio
task may be possible after class hierarchy and grouped split design are
resolved. A joint AV task needs additional visual labels or a preregistered
test showing complementary necessity with suitable unimodal controls.

Raw video has not been inspected for people or license plates. Privacy and
product-specific media handling remain REVIEW even though the dataset license
is permissive. No source rows are approved for model training by this document.

## Next gate

Do not use the official TRAIN/VALIDATION split to select a model. If MAVD is
revisited, first build and hash a new development partition grouped by
site-day (or stronger recording-session identity), reconcile the ontology,
and complete video privacy review. Keep the held-out official TEST label
payload sealed. Admit an AV task only if its target is truth-bearing for both
inputs or the preregistered counterfactual controls establish that both are
needed; media co-occurrence alone is insufficient.
