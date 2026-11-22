# Hoi! Dataverse metadata capacity reconciliation — 2026-11-22

## Finding

A fresh read-only query of the official version-1 file-list metadata resolves
the *outer file-list* size for `bathroom_1/gripper`, but not the expanded size
or working storage needed to process it.

| Scope | Outer metadata rows | Listed outer bytes | ZIP files / listed compressed bytes |
|---|---:|---:|---:|
| Whole Hoi! version 1 | 4,769 | 3,186,202,640,224 | 3,181 / 3,185,687,461,273 |
| `bathroom_1/gripper` path prefix | 57 | 27,417,208,876 | 39 / 27,413,439,341 |

The path-prefix calculation uses `directoryLabel` equal to
`bathroom_1/gripper` or beginning with `bathroom_1/gripper/`. It sums the
Dataverse `dataFile.filesize` values for each matching outer file. ZIP bytes
are compressed archive sizes. They are not uncompressed member sizes or
estimated peak working storage.

The previous 2026-11-22 source-screen receipt reported 1,421 files and
1,274,184,763,539 bytes for this subtree. Those figures are **not reproduced
by the outer Dataverse file-list rows**. They could refer to expanded contents,
but no reproducible source for that interpretation was found in this audit;
record them as unverified rather than silently substituting one number for the
other. This does not establish that a selected archive subset is feasible: a
bounded acquisition/extraction plan and expanded-size evidence are still
missing.

## Grouping limitation

The current version-1 listing has 21 non-empty top-level `directoryLabel`
roots (22 distinct roots if the one root-level entry is counted). The publisher
reports 38 environments. Directory labels therefore do not recover all
environment or participant identities. The existing filename-derived
force/telemetry key union remains 198 candidate `(environment, index)` keys;
these are not proven independent object/session groups and remain below the
provisional 200-group gate.

## Reproduction and scope

- Dataset version: `1`
- Dataset metadata endpoint response: 4,269,526 bytes, SHA-256
  `a99356070fd2215cc86870971055bf194e43ee40209d454fb8513998f61f6f6a`
- Versioned file-list endpoint response: 4,258,029 bytes, SHA-256
  `ee1b129467d2b68ebc32aafef099ba351406c7ddc0bb8ff36418a8e7406e4d51`
- Both endpoints report 4,769 file rows and 3,186,202,640,224 listed bytes.
- Command: `python scripts/audit_hoi_source_metadata.py`
- Tests: `python -m pytest tests/test_hoi_source_metadata.py -q` — 4 passed.
- Metadata parser/Ruff: passed.
- No payload endpoint, archive, annotation, model, or GPU was accessed.

The metadata summary script now reports directory-prefix file counts and outer
bytes separately from compressed ZIP bytes. See the
[base source screen](R2_HOI_FORCE_VIDEO_SOURCE_SCREEN_20261122.md) and the
[machine-readable reconciliation](../../manifests/pretrained_reuse/hoi-force-video-source-screen-20261122.json).
