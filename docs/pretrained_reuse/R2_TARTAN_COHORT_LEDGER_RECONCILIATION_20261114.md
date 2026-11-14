# TartanAviation cohort exposure ledger reconciliation — 2026-11-14

## Result

The fail-closed cohort usage checker and the existing Perception Test ledger
entry were brought onto the current PR #15 roadmap head (`bfb13bf`). The
TartanAviation official source and both previously screened metadata
derivatives are now registered as `INCOMPLETE` histories:

| Source | Revision | Group identity | Known metadata exposure |
|---|---|---|---:|
| `castacks/TartanAviation` | `077a04abac1e50953c45b72524edd79e3a21d9bd` | parent clip | 40,899 groups |
| `twangodev/tartanaviation-atc-labels` | `0775355c388aae64335522f40bff9fc5129b2922` | parent clip | 40,899 groups |
| `twangodev/tartanaviation-atc-adsb-utterances` | `bc588362d2f877e6ba0910fe7b6e035649d43918` | parent clip | 40,899 groups |

Every proposed cohort from these revisions fails closed with
`BLOCKED_INCOMPLETE_HISTORY`. The stored ledger records the known exposure
count and pins the hash-only roster, but does not inline 122,697 digests into
Git. The roster remains outside the repository at
`D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adbscreen-20261114-attempt-01\group_exposure_roster-v2.json`;
its SHA-256 is
`6426469a01127947190f570ed0776a77cbe61d73c7b59e44520500beafa4c9f3`.
That roster derives from 184 hash-verified Parquet shards and contains only
source/revision-scoped parent-clip digests, not raw IDs, transcript text,
callsigns, labels, or media.

This is an exposure-control reconciliation, not a task admission. The source
screen remains `REVIEW_NOT_ADMITTED`: available labels are insufficient and
partly circular, independent grouping is not established, and both-input
necessity is unproven. No source payload, label text, media, model, validation
labels, or sealed audit was opened for this change; no training or GPU work
was performed.

## Implementation and verification

- Source checker and original Perception Test ledger entry: commit
  `5bd71a11ae726d39f17f81464c811c985bb5f2cf`, copied without edits.
- Added regression checks for all three Tartan source/revision pairs. A
  synthetic unseen group is blocked for each because history is incomplete.
- `python -m pytest tests/test_check_cohort_usage.py -q`: 11 passed.
- Full repository CPU suite: 2,907 passed, 11 skipped.
- `ruff check .`: passed.
- CI-equivalent model/dataset manifest and catalog validation commands:
  passed (training catalogs: 9 and 8 sources; evaluation catalogs: 11 and 10
  sources; example dataset-manifest audit passed).
- `git diff --check`: passed.

The next task-quality step remains a fresh source/task admission gate. Do not
reuse Tartan records unless a complete history audit, non-circular verified
labels, adequate independent groups, and a valid both-input control design
are established first.
