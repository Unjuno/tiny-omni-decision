# ZOD cross-branch evidence reconciliation (2026-10-28)

Status: **promising camera/LiDAR association task, not admitted**. Read-only
reconciliation of open Draft PR #35; no files in its worktree, source data,
or artifacts were modified.

## New evidence

PR #35, head `3160975fac62e1c1cb79ad91dc94b0f3b7bf2bcb`, adds a pinned official
SDK schema check: one object annotation can contain a camera 2D box and a
LiDAR-frame 3D box joined by the same `annotation_uuid`. This makes a
candidate-matching task schema-feasible. It does not demonstrate that a model
needs both modalities or general driving decision quality.

The official download page exposes both an access-request/CLI route and an
Academic Torrents route; whether the torrent path waives the request remains
unresolved. The official license is CC BY-SA 4.0 with attribution,
share-alike, public-notice, and non-military terms. How those terms apply to a
trained model has not been legally assessed. A mini/info/annotation archive
set is listed at about 2.98 GB compressed; this is only a descriptor claim,
not downloaded payload or extracted-size evidence.

## Candidate task and remaining gates

A narrow supplied-option task could match a marked 2D object crop to one of
several candidate 3D LiDAR boxes using shared annotation UUIDs. A valid
comparison would need matched image-only and LiDAR-only baselines, plausible
same-class/range distractors, shuffle controls, and train/validation/audit
groups separated by sequence and parent drive/session. Existing annotations
are on central sequence keyframes; temporal generalization is not established.
Frozen-encoder pretraining overlap, usable pair counts, split groups, and
share-alike implications remain unverified.

Do not download the ZOD payload, request access, train, or select a model from
this task until these gates are resolved and local storage is sufficient. The
earlier 2026-10-21 source screen remains historical; this note supersedes its
statement that camera/LiDAR object co-reference was unestablished, while
retaining its unresolved access, rights, grouping, and task-necessity gates.

## Evidence

- [Open Draft PR #35](https://github.com/Unjuno/tiny-omni-decision/pull/35),
  head `3160975fac62e1c1cb79ad91dc94b0f3b7bf2bcb`; its two CI checks passed.
- [PR report: schema and torrent follow-up](https://github.com/Unjuno/tiny-omni-decision/blob/3160975fac62e1c1cb79ad91dc94b0f3b7bf2bcb/docs/pretrained_reuse/R2_ZOD_SCHEMA_TORRENT_FOLLOWUP_20261028.md).
- [Official ZOD access page](https://zod.zenseact.com/download/),
  [license](https://zod.zenseact.com/license/),
  [Frames](https://zod.zenseact.com/frames/) and
  [Sequences](https://zod.zenseact.com/sequences/).
