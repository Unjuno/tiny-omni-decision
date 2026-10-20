# R0 live-state audit — 2026-10-20

This is a read-only state snapshot for the pretrained-reuse roadmap. It does
not modify any legacy branch, PR, run, dataset, or checkpoint. The only working
branch changed for this task is the dedicated `codex/r2-perception-fresh-holdout`
branch.

## Git and roadmap state

- Repository: `Unjuno/tiny-omni-decision`.
- Latest roadmap branch head observed: `43974fb2f6b070e0182465edca12d061b65ed8dd`.
- PR #15 is open and draft; head `43974fb2f6b070e0182465edca12d061b65ed8dd`,
  base `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- GitHub REST listed open PRs #9–#15. PR #9–#14 remain open; no merge, close,
  rebase, or write was performed on them.
- The local repository exposes 21 worktrees. The OneDrive user worktree and
  all other worktrees were left untouched. The active dedicated worktree is
  `C:\CodexWorktrees\r2-perception-fresh-holdout`, branch
  `codex/r2-perception-fresh-holdout`, clean at the start of this audit.
- Dedicated branch commits from this work are pushed through `52a9b37`;
  CI for `52a9b37add6562cb3a8190e6b2d810400be63de8` passed.

## Active processes and compute

At 2026-10-20 08:19 UTC:

- Perception Test media acquisition PID 27560 was alive, with a pinned
  `RUNNING` receipt at 117/200 videos. Its HTTPS connection was active on the
  prior poll. The process and its external output were preserved.
- Orbitune supervisor/training processes remained live. RTX 3080 Laptop GPU
  reported 40% utilization and 15,821/16,384 MiB allocated. No training or
  feature-extraction GPU process for this branch was started.
- Paid/cloud compute used: $0.

## Relevant pinned artifacts

- Fresh TRAIN holdout plan SHA-256:
  `887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3`.
- Media acquisition is incomplete; do not treat its current receipt as a
  complete corpus.
- Partial CPU PTS audit on 101 videos / 246 events passed, with zero incomplete
  audio intervals, zero right-censored events, and 1,692 unique exact media
  hashes including prior artifacts. This is an intermediate integrity check,
  not a model-quality result or authorization to extract features.
- Existing fixed AV checkpoint/result pairs were rehashed and matched their
  receipts. They remain unchanged and are only intended for the frozen
  same-checkpoint comparison after the complete media/PTS gates pass.

## Open roadmap gates

The current activity is a narrow R2 fresh-TRAIN transfer check. It does not
close R0's deployment-device/task acceptance contract, R2's direct modality-
native and truth-bearing joint-task requirements, or the later R3–R8 gates.
No product-quality or 95–98% claim follows from the partial audit.
