# R0 artifact-capacity recheck — 2026-11-03 22:55 JST

## Live repository state

- Target worktree: `C:\CodexWorktrees\r2-perception-control-freezer-20261031`.
- Branch `docs/pretrained-reuse-roadmap-20261010` and `origin` both point to
  `d1f640fdf6c3059b8ab97dd3cd34fdbb7d76a036`.
- PR #15 remains open and Draft, with the same head. The CI run for this commit
  (`41648521440`) completed successfully.
- Two untracked PhysicalAI split manifests are pre-existing user work and were
  left untouched.

## Storage and process state

- C: free space was **1,913,630,720 bytes** at the adjacent 22:55 JST read,
  below the extractor's fixed 2 GiB reserve by 233,852,928 bytes.
- The fixed RoboMNIST AV-match cache still has **1,778 / 1,780** group
  directories and `run.json` status `INCOMPLETE`, last updated 22:26:52 JST.
  Its last error remains `disk reserve reached`; no Python, FFmpeg, or download
  process was active during this snapshot.
- User-owned Quartus process PID 28280 was active. This report does not infer
  that it caused the observed capacity change and does not alter its build.
- The two exact-prefix partials described in the pass-2 cleanup receipt remain
  present because automatic action review rejected their removal. No alternate
  cleanup path was attempted; bytes freed by that attempt: zero.

## Disposition

Do not lower the extractor reserve, omit the two validation groups, or change
the frozen split/configuration. Resume only after stable capacity is restored
and rechecked. Keep the run and all other partials, source media, features,
checkpoints, and receipts unchanged.
