# R0 live-state recheck (2026-10-11, Asia/Tokyo)

This is a read-only point-in-time refresh of the R0 inventory. It supplements,
and does not rewrite, the historical snapshots. No other worktree, running
process, dataset, checkpoint, or legacy artifact was changed.

## Repository, branch, and worktrees

- Repository: `Unjuno/tiny-omni-decision`.
- Active branch: `docs/pretrained-reuse-roadmap-20261010`.
- Remote-tracking HEAD at refresh: `6dee879220398008117b8075a55c7e421867b8e8`.
- `git fetch origin` completed; branch and `origin/docs/pretrained-reuse-roadmap-20261010`
  were aligned before this refresh's candidate-screen edits.
- Main remains `09246d7b1821662a6b1eba47fc08c43f9dc3863c`; no merge was done.
- The repository has independent worktrees for Teacher v1, video long-budget,
  ternary recovery, QAT groupwise diagnostics, and PR #14 review. They remain
  at their separate heads. The current worktree list is available from
  `git worktree list`; no other worktree was checked out or edited.

## Open pull requests, observed through GitHub API

| PR | State | Head → base | Head SHA |
|---|---|---|---|
| #15 | Draft, open | `docs/pretrained-reuse-roadmap-20261010` → `main` | `6dee879220398008117b8075a55c7e421867b8e8` |
| #14 | Draft, open | `fix/student-qat-training-handoff` → `codex/ternary-student-recovery` | `c3391a0c689d69df3300686d9c287546b741e7c8` |
| #13 | Open | `feat/ternary-student-recovery` → `main` | `6a343af31b2d064de302dc9d93eeca17b27dcadd` |
| #12 | Draft, open | `codex/postquant-teacher-cache` → `codex/postquant-numerical-core` | `c79a50bd937de9266fb1277efd68fd24b3989a61` |
| #11 | Draft, open | `codex/postquant-numerical-core` → `codex/postquant-worker-entrypoints` | `8e39f7a8088d5fb8a77c93ee763b5b01484a63b5` |
| #10 | Draft, open | `codex/postquant-worker-entrypoints` → `codex/roadmap-compression-video-specialization` | `1fbe0adad7e9307baabbdc9387b8e27819b78932` |
| #9 | Draft, open | `codex/teacher-quality-v1` → `main` | `8bf351c9b2537e03780d55edf12c45ad09819134` |

PR #13 (`feat/ternary-student-recovery`) is distinct from PR #14's fix branch
and base relationship. All were left untouched; none was merged or closed.
PR #15 CI runs #476 and #477 passed at the observed head above. Later candidate
screen changes in this working tree have not yet been pushed at this snapshot.

## Processes and compute

The RTX 3080 Laptop GPU reported 16 GiB VRAM, 11 MiB used, 0% utilization.
The only listed GPU process was the Codex desktop application. No model
training or evaluation process was running.

| Process | Observed command / state | Action |
|---|---|---|
| PID 26456, Python, started 2026-10-05 17:10 JST | `manage.py runserver 127.0.0.1:8000` | Left running |
| PID 66896, Python, started 2026-10-08 12:16 JST | Second process with same Django `runserver` command | Left running |
| PID 78668 and child Git/credential processes, started 2026-10-10 23:12 JST | `git ls-remote https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L3-v2 HEAD` | Left running; no kill/restart attempted |

No cloud compute or paid services were used. No GPU work was started in this
refresh.

## Relevant external artifacts

- VideoPrism-B checkpoint remains at
  `C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz`, 457,475,296
  bytes, SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Audio runtime bundle v2 remains at
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-multiquery-runtime-bundle-v2`;
  manifest SHA-256 `574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`.
- Its same-validation-ID runtime replay report is
  `C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-runtime-validation-v1\report.json`,
  SHA-256 `adb75cbdafd47099655618523d883ada0def0d5f1fca34f3503a8eb0bce2e8f8`;
  2,560 audio decisions, zero prediction mismatches at encoder batch size 8.
- Existing V-JEPA and CLEVRER feature caches/reports remain under the R1/R2
  artifact roots; no sealed audit was inspected.

## Open control gate

ROADMAP R0 still lacks a user-confirmed target device, per-task quality floors
and confidence bounds, and time/cost limits. The 100–200M parameter and
<=200 MB package figures remain roadmap targets, not approved product release
criteria. Continue bounded local frozen-feature work; do not make a product
selection, promotion, unsealing, or paid-compute decision from this snapshot.
