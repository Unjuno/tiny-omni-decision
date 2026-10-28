# R0/R2 open-PR reconciliation — 2026-10-28

This read-only PR review supplements the roadmap's earlier branch inventory.
It is intended to prevent repeating source screens or exposed development
evaluations. No PR was changed, merged, or closed.

## GitHub state

Public GitHub API state was checked on 2026-10-28. PRs #31–#36 are all
Open/Draft. The base and head are separate; none has been merged into the
roadmap branch.

| PR | Base | Head SHA | Current finding |
| --- | --- | --- | --- |
| #31 | codex/r2-droid-expanded-sites-20261028 | aefb1a1a3cd09959e73dfac79dbf319fce63c367 | Six sampled shards contain no new collection site. |
| #32 | codex/r2-droid-site-universe-audit-20261028 | a8db60d262b9655cb00897f26cdec9ee6a177070 | On 137 held-out episode rows from 14 sites, VideoPrism plus sensor is weaker than sensor-only. |
| #33 | codex/r2-droid-temporal-transfer-20261028 | b8245103d221daea225f9666862f6b149462aa1b | Paired site bootstrap: Accuracy difference −2.92 pp (95% CI −8.08 to +0.97); NLL and Brier are worse for the joint readout. |
| #34 | codex/r2-droid-later-shard-uncertainty-20261028 | 2a6a022f9974c2b0aec79da759b4d1fd0d27b57a | GOOSE image/point-cloud class-instance correspondence varies substantially across 15 train scenarios; not admitted for supervision. |
| #35 | docs/pretrained-reuse-roadmap-20261010 | 3160975fac62e1c1cb79ad91dc94b0f3b7bf2bcb | ZOD has a plausible camera/LiDAR co-reference task, but data-access permission, drive-level grouping, overlap and both-input benefit remain unresolved. |
| #36 | docs/pretrained-reuse-roadmap-20261010 | 57aa474d9f15b2f70d9f9dc6fb441e69d377ba99 | FZI-AURA sample is quarantined because the SDK's required acceptance flow was bypassed. DAVE is held because underlying components include non-commercial/agreement-bound data. |

## Admission decision

None of these PRs adds an R2 task approved for product-aligned training or
quality selection. DROID's later-shard result does not justify another fit on
those exposed groups. GOOSE's labels are too inconsistent for the proposed
target. ZOD requires resolving its provider access route and sequence/drive
independence before payload use. The FZI-AURA sample must not be used while
access is unresolved; no further reads or processing are authorized by its
screen. DAVE's apparent MIT dataset-card label does not clear its component
media or annotation restrictions.

The other recently reviewed natural AV candidates remain held by the reasons
already recorded in the roadmap: QIVD's binding non-commercial agreement,
AVQA's source/media restrictions, and unresolved component rights for
UnAV-100. No new source in this PR set closes R2-M's synchronized, adequately
grouped, both-input-required task gate. Architecture B/C or additional
multimodal fitting is therefore not justified by these PRs.

The separate one-row ADL CPU feature smoke is documented in
R2_ADL_AUDIO_VIBRATION_FEATURE_RUNTIME_20261028.md. That smoke uses only a fixed
training row, measures no Decision quality, and does not change ADL's
12-train/2-validation participant limitation.

All prior worktrees, PR branches, external data, model weights and experiment
artifacts were preserved. No sealed audit was opened or re-evaluated. The
existing local user worktree was not modified.
