# R0 live recheck — 2026-11-28 16:23 JST

## Scope

Read-only refresh of the remote roadmap PR, main, selected local worktrees,
CI status, local storage, and GPU occupancy. No live process, training run,
checkpoint, corpus, or user worktree was changed. No training or model
evaluation was started.

## Remote state

- `main`: `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- PR #15 is **open / draft**, base `main`, head
  `docs/pretrained-reuse-roadmap-20261010` at
  `583853d6bea34f3da424c8b8e7adfcd8e8e84af6`.
- The two GitHub Actions `test` check runs attached to the PR head both
  completed successfully on 2026-11-26 at 00:20 UTC. This is CI evidence for
  that commit only, not evidence that the open research gates are complete.
- The independent data-audit branch is
  `codex/pretrained-reuse-r2-data-audit-20261128` at
  `0d3d79c0c1d1c41edff554ab176cd5a6f3f112a0`, matching its remote ref and
  clean at the time of inspection.

The GitHub pull-request API was readable without authentication. `gh pr view`
could not authenticate, so review-comment and full PR conversation status were
not rechecked in this snapshot.

## Worktree preservation

- The PR-head worktree at
  `C:\CodexWorktrees\r2-perception-control-freezer-20261031` is on
  `docs/pretrained-reuse-roadmap-20261010`, local HEAD
  `ea014431b163f69176c52f5df8944f2dc7f7583f`, 36 commits behind the remote
  head. It has two untracked PhysicalAI split manifests. Their contents were
  not read or changed; do not fast-forward or clean this worktree without
  first preserving those user files.
- The OneDrive checkout remains on
  `codex/teacher-quality-clean-dev-v1` with pre-existing modifications and
  untracked files. It was not used for edits.

## Local resource snapshot

- RTX 3080 Laptop: 16,384 MiB total, 8,270 MiB reported used, 16% utilization.
  `nvidia-smi` identified a separate Python compute process. Process metadata
  associated it with the existing Orbitune work; it was left running.
- C: had 25,415,192,576 bytes free (23.67 GiB).
- No GPU work was started. Existing process scheduling and stop behavior were
  not tested or altered.

## Decision and next gate

The BERT-family reports already show task-specific signal, including a
89.94% one-time BANKING77 test result for frozen MiniLM-L3, but that task has
known label ambiguity and no user/session provenance. Japanese BERT-base also
exceeds the provisional whole-model parameter budget in the four-tower
estimate. No exposed validation or publisher test should be reused for tuning.

The immediate bottleneck is therefore an admitted, source-disjoint task/data
cohort—not a shortage of additional BERT architectures. Continue R2 with data
quality and source-admission work; keep current exposed BERT results as fixed
references. Start no GPU experiment until the Orbitune process is confirmed
terminal and the task/data gate is explicit.

## Evidence and limits

This snapshot used `git ls-remote`, the unauthenticated GitHub pull/check-runs
API, `git worktree list`, `git status`, `nvidia-smi`, Windows process metadata,
and the C: logical-disk free-space counter. The snapshot is not a complete
inventory of every remote branch, dataset, artifact, or PR discussion; use the
preserved inventory and source-specific reports for those details.
