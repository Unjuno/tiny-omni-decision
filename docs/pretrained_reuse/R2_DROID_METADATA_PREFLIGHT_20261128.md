# R2 DROID metadata-only preflight — 2026-11-28

## Decision

DROID is a **screened candidate, not an admitted training or evaluation source**.
The official metadata index contains both success- and failure-path records in
all 13 labs. A fixed stratified sample of 520 episode metadata files had
explicit `success` values matching their directory status, usable task text,
and positive trajectory lengths. Twenty-four sampled task hashes occurred in
both path statuses. This is enough to justify a bounded label/task and split
feasibility review; it is not evidence that task-matched episodes are
independent, that the labels are visually observable, or that video improves
over task/state metadata.

No trajectory HDF5, video, SVO, checkpoint, GPU, validation/test partition, or
sealed data was accessed. No model was trained. Do not start a dataset-scale
download or claim Decision quality from this preflight.

## Pinned source and rights

- Official raw source: `gs://gresearch/robotics/droid_raw/1.0.1/`.
- Official repository revision inspected: `33ae6a67274f36d2e29525b86f23a56616ef43a7`.
- The DROID paper states the full dataset is released under CC BY 4.0:
  <https://autolab.berkeley.edu/assets/publications/media/2024-RSS-DROID.pdf>.
- The official repository documents the raw and RLDS layouts and their very
  large sizes: <https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/docs/the-droid-dataset.md>.
- The pinned collector initially writes under `failure/`, moves to
  `success/` when `controller_info["success"]` is true, and allows operator
  relabeling. Therefore folder names alone are not targets; require the
  explicit metadata boolean, and quarantine missing or contradictory values.
  See <https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/user_interface/data_collector.py>.
- The pinned schema exposes `success`, `scene_id`, `building`, `user_id`, and
  `current_task`; scene/building fields can be missing or defaulted. See
  <https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/postprocessing/schema.py>.

The paper-level CC BY 4.0 statement is a useful rights signal, but source-video
privacy, per-record provenance, and product redistribution implications still
need review before media use or product claims.

## Read-only metadata inventory

The public GCS object index listed 74,896 metadata JSON objects across 13 labs;
each lab had entries under both success and failure paths. A separate
success-only `manifest.json` listed 38,250 paths and cannot define the failure
cohort. These counts describe index objects, not verified independent
examples.

The inventory metadata object was generation `1710761242404629`, 5,147,314
bytes, MD5 `eT761Ed17j8hvt4kjtY8zw==`; SHA-256 of downloaded bytes:
`f540f0a86e791d11588670f82e8045b54ae79959a8104aa6781087486f65a5d2`.
The aggregate-annotations object was generation `1711040208989663`,
12,088,718 bytes, MD5 `kY4KXWAXpEb9RP0QCJT3sA==`; SHA-256:
`42ed672ee974bfcb15752dc0768dd66a47e01ea3eebfdb74d8d5e468b37b2b6d`.
The aggregate annotation is success-side and was not treated as a balanced
classification corpus.

A deterministic sample selected at most 20 metadata objects per lab and path
status, with seed `droid-r2-metadata-preflight-20261128-v1`:

| Check | Observed |
|---|---:|
| Episode metadata fetched | 520 |
| Explicit `success=true` / `false` | 260 / 260 |
| Path/boolean mismatches | 0 / 520 |
| Empty task strings | 0 / 520 |
| Positive trajectory length | 520 / 520 |
| Missing `building` / `scene_id` / `user_id` | 5 / 5 / 0 |
| Task hashes appearing in both path statuses | 24 |
| Media, HDF5, SVO, or model payloads fetched | 0 |

The per-lab/path index inventory SHA-256 is
`573cd498b74cefb3a07e73878a49abea9230768f52dd9621030e1eb4d5ff1a58`; the
sampled record-content-hash list SHA-256 is
`30a664430503dc3b3a9e5d6692da2de7faf2f60926c81ffee7ffebf3c785fa45`.
The local machine-readable summary is
`C:\CodexArtifacts\pretrained-reuse-r2\droid-metadata-screen-20261128\metadata-audit-summary.json`
(9,446 bytes, SHA-256
`0832f3906602dba3b53a8352ed7fd2c638db2380813dfe0f6036aeb1e215e446`).
The summary contains hashes and aggregate counts, not raw task text or object
IDs. The audit implementation is `scripts/audit_droid_gcs_metadata.py` with
three focused tests in `tests/test_audit_droid_gcs_metadata.py`.

## Task-semantic sample review

The same immutable 520-record sample was re-fetched by object generation and
MD5 and reviewed without retaining record IDs. The 24 cross-status task groups
include repeated task families such as putting/removing objects in containers
(58 success / 53 failure examples across 47 / 39 sampled scene groups), moving
objects to new positions/orientations (33 / 35 across 28 / 30 groups), hinged
objects (20 / 19 across 15 / 14), and folding/spreading or clumping objects
(13 / 12 across 13 / 12). Other recurring groups include slidable objects,
hanging/unhanging, buttons, lids, and pouring. This gives a concrete reason to
continue task-level split and label checks: multiple sampled scene groups have
both outcome labels for many instruction families. These are stratified
sample counts, not population estimates or independently verified episodes.

The short-list artifact is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\droid-metadata-screen-20261128\task-semantics-shortlist.txt`,
4,186 bytes, SHA-256
`55731791f935213d8d9cbd0bf652c1413d5e43ef36e153e4c6c0c65bfc423794`.
It contains task descriptions and sample counts only, with no source IDs or
media. The descriptions often express broad task families rather than exact
actions, so exact task-hash overlap does not prove equivalent difficulty or
label comparability.

## Data gates before any model run

1. Expand metadata-only inspection for task families with success and failure
   support. Verify that `success` represents the intended outcome and exclude
   aborted/incomplete or operator-ambiguous records. Never use path strings as
   model features.
2. Build source-disjoint groups using lab, building, scene, user, task, and
   robot/session identity where available. Report missing grouping fields and
   reject any candidate split with group or normalized-content overlap.
3. Define a future-outcome Decision using pre-outcome video, action/state, and
   task context. Compare video+context with context-only, state-only, and
   video-only controls on the same source-disjoint IDs. Reject the task if the
   permitted context predicts outcomes without video or if post-outcome cues
   reveal the target. Exclude path, filenames, and target-leaking metadata.
4. Verify the exact media license/provenance and perform a small rights-cleared
   train-only media/runtime check in a new output location. Do not download the
   full DROID corpus (the official docs describe terabyte-scale bundles).
5. Freeze a fresh validation set by scene/lab/user before fitting a readout.
   Only after the above checks pass can DROID advance to an R2 frozen-feature
   Decision experiment.

