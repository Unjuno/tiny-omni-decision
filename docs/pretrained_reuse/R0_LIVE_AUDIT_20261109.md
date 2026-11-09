# R0 live repository and resource audit — 2026-11-09

## Scope and method

Read-only snapshot to refresh the Oct 29 repository audit before further R2
work. Queried the public GitHub REST API for open pull requests, inspected local
Git worktree/remote-branch metadata, process names, free disk, and GPU telemetry.
No process was stopped, no model/data/checkpoint was read or modified, no
training or model inference was started, and no worktree was cleaned.

Snapshot time: **2026-11-09 22:57 JST**. The local repository was fetched from
`origin`; the roadmap branch was at `e07e049ff4387484fb754e0fff963631f3cc7c85`.
The PR #45 worktree was clean at `bab97ff8ee32b3cbca716c792f26520e01691130`,
matching its remote head. The user-facing OneDrive worktree is on a separate
Teacher branch and contains uncommitted changes; it was left untouched.

## Live inventory

| Item | Observed state |
|---|---|
| Open pull requests | 37, from unauthenticated public GitHub REST API |
| Local Git worktrees | 97 |
| Remote-tracking branches | 131 |
| Dirty worktrees | 9; paths and contents preserved |
| RTX 3080 Laptop GPU | 0 / 16,384 MiB, 0% utilization |
| C: free space | 7,869,169,664 bytes (about 7.33 GiB) |
| Matching Python / Torch / Orbitune / Quartus / Questa processes | None observed |
| GitHub CLI authentication | Unavailable; public REST API used instead |

The live PR snapshot identifies **PR #15** as Draft/Open, head
`docs/pretrained-reuse-roadmap-20261010` at `e07e049f`, base `main`. **PR #45**
is Draft/Open, head `codex/r2-train-only-temporal-cv-20261104` at `bab97ff8`,
base `docs/pretrained-reuse-roadmap-20261010`. Both queried commits have two
completed successful GitHub Actions `test` check-runs. The separate aggregate
commit-status endpoint returned `pending` with zero status contexts, so only
the check-run conclusions are asserted here.

Raw snapshots are retained outside Git:

- `C:\CodexArtifacts\pretrained-reuse-r2\r0-open-pr-snapshot-20261109.json` —
  SHA-256 `ac87d3608fa1c483af425d51dc266e839b1b443e097f9f26607b6f3d9a0dae24`.
- `C:\CodexArtifacts\pretrained-reuse-r2\r0-worktree-snapshot-20261109.json` —
  SHA-256 `77658981940e8e08ea025799bbeecc632c6c505c8587b1c6ec9e2cd3bc7247009`.

## R1/R2 decision from current evidence

The existing R1 record already verifies the official VideoPrism-B checkpoint
and runtime path: 114,365,184 parameters, pinned checkpoint/model revisions,
CC-BY-4.0/Apache-2.0 upstream evidence, and finite local 8-frame execution.
Repeating that load or rights check would duplicate completed work. V-JEPA 2.1
remains `REVIEW`; the exact checkpoint grant is unresolved, so it is not a
candidate for packaging or promotion.

R2 still has **no admitted new real joint task** for another fit. The Nov 4
admission matrix and Nov 5 source screens reject or hold the recent candidates
because of exposed validation, weak calibration/transfer, unresolved source
rights, insufficient independent groups, unresolved scene identity, or lack of
evidence that both streams are necessary. PR #45's TRAIN-only diagnostics also
show overfitting on the same-repetition AV match, no benefit from the ordered
LSTM over pooled features, and weak condition transfer on the digit task. These
are task-scoped negative development results, not a general representation
limit.

The existing frozen-feature Path A code and modality-specific runtime probes
are real implementations, but the R2 product-quality gate is still open. The
next quality evaluation must use a newly admitted rights-cleared task and
independent groups; the currently cached/exposed validation sets are not a
substitute. No R3 shared-backbone or merge experiment is justified by this
snapshot.

### EgoViz audit-set gate recheck

The official [EgoViz-120 dataset card](https://huggingface.co/datasets/humaidtech/EgoViz-120)
currently documents CC-BY-4.0, 180 capture-session `group_id`s, and says clips
within a session share a worker/site/lighting. Its documented clip metadata
does not expose a cross-session participant identifier or guarantee that a
worker appears in only one session. The reserved 20-session split therefore
cannot yet establish participant-independent evaluation. The already-reported
69-session transfer has no matched video-only, sensor-only, or metadata-only
control on that cohort; its timestamp-matching target is narrower than a
natural action decision, and the GRU endpoint's NLL/ECE were poor. Keep the
reserved sessions unopened until a task-level candidate and controls are
frozen; even then, report the result only as session-held-out evidence unless
participant independence is verified.

## Outcome

R0's live-state picture is refreshed and auditable. No experiment was launched
because the data/task admission gate, rather than local GPU availability, is
the active constraint. Existing worktrees, PRs, and artifacts remain intact.
