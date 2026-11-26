# R0 open pull request and dependency review (2026-10-11)

This is a read-only inventory and direction-fit review of the open legacy PRs
#9–#14, requested by the R0 preservation gate. It does not approve, merge,
close, rebase, or otherwise change any PR, branch, experiment, or artifact.

## Snapshot and evidence

GitHub REST metadata and file lists were read for each PR on 2026-10-11. The
local worktree was `docs/pretrained-reuse-roadmap-20261010` at
`06519243116a0d5b307c34cff9a7a85632ce8cab`, clean and equal to its remote
branch. Current `main` is `09246d7b1821662a6b1eba47fc08c43f9dc3863c`. The local
RTX 3080 Laptop was idle at 0% utilization with 11 MiB reported in use; the
only Python processes were two Django development servers. No training
process was found.

All six PRs were open at the time of inspection. GitHub returned no submitted
reviews or inline review comments for them. Check-run results were confirmed
as successful for PR #9, #10, #11, and #12 heads. Check-run status for #13 and
#14 is **UNKNOWN**: unauthenticated API rate limiting interrupted those
lookups, and the separate combined-status endpoint returned no status rows.
Do not interpret an empty status list as a passing check.

## PR inventory and relationship to R0–R8

| PR | Head and base at inspection | Scope and dependency | R0–R8 disposition |
|---|---|---|---|
| [#9](https://github.com/Unjuno/tiny-omni-decision/pull/9) Teacher v1 | `codex/teacher-quality-v1` at `8bf351c9`; base `main` at `498f557e` (older than current main); GitHub marked it not mergeable at inspection. | Leakage-safe Teacher v1 corpus/split gates, training configs and implementation, manifests, tests, and the Video Teacher v2 design spec. It is an important historical experiment and source of split-integrity evidence, but it implements the older Teacher training track rather than the frozen-pretrained Path A runtime. | Preserve the branch, Teacher v1 artifacts, and PR. Reuse its factual split and training evidence where relevant; do not merge it as part of R0–R8 or use its old validation/audit numbers as new Path A evidence. |
| [#10](https://github.com/Unjuno/tiny-omni-decision/pull/10) postquant workers | `codex/postquant-worker-entrypoints` at `1fbe0ada`; base `codex/roadmap-compression-video-specialization` at `91d7867f`. | Adds guarded postquant worker entrypoints and launch preflight. It does not contain numerical backend/model training. | Separate ternary/recovery track; not a dependency of R0–R8. Preserve without launching workers. |
| [#11](https://github.com/Unjuno/tiny-omni-decision/pull/11) postquant numerical core | `codex/postquant-numerical-core` at `8e39f7a8`; base is exactly PR #10 head `1fbe0ada`. | Adds reference ternary numerics, losses, policy, and CPU checker on top of #10. No real model execution. | Stacked dependency: #10 → #11. Preserve the stack; do not treat the reference quantizer as a deployed low-bit kernel. |
| [#12](https://github.com/Unjuno/tiny-omni-decision/pull/12) postquant teacher cache | `codex/postquant-teacher-cache` at `c79a50bd`; base is exactly PR #11 head `8e39f7a8`. | Adds a strict teacher-cache boundary on top of #11. It does not load the Teacher or generate logits. | Stacked dependency: #10 → #11 → #12. No dependency on the frozen-feature Path A work. |
| [#13](https://github.com/Unjuno/tiny-omni-decision/pull/13) ternary-first student recovery | `feat/ternary-student-recovery` at `6a343af3`; base current `main` at `09246d7b`. | Implements the EmbeddingGemma 2 ternary-first student and conditional Recovery LoRA path. This is the separate `feat/` branch, not `codex/ternary-student-recovery`. | Preserve as the legacy compression experiment. Its architecture and goals differ from pretrained world-representation reuse; do not merge or promote it under this roadmap. |
| [#14](https://github.com/Unjuno/tiny-omni-decision/pull/14) QAT handoff/performance | `fix/student-qat-training-handoff` at `c3391a0c`; base branch `codex/ternary-student-recovery` at `acd20f04`. | Optimizes and changes handoff semantics for the QAT trainer on the `codex/` recovery lineage. Its own description explicitly distinguishes this from PR #13's `feat/ternary-student-recovery` reference CLI. | Preserve separately from #13. Do not conflate the two branches, alter a run, or use `best-shadow` as an exact optimizer resume. It is not a Path A dependency. |

The dependency graph observed from PR base/head metadata is:

```text
PR #10 (base: codex/roadmap-compression-video-specialization)
  └─ PR #11
       └─ PR #12

PR #9: independent, base main at 498f557e
PR #13: independent, base current main at 09246d7b
PR #14: base codex/ternary-student-recovery at acd20f04
```

PR #14 is **not** based on PR #13's `feat/ternary-student-recovery` head. The
similarly named branches must remain distinct in future status checks and
handoffs.

## Action

No PR or worktree was modified. Keep the PRs open and their branches/artifacts
preserved. The new R0–R8 branch remains the active research direction; it must
not silently absorb the legacy Teacher or ternary-first work. Revisit an
individual PR only through an explicitly scoped review or when a later R0–R8
gate has a concrete dependency on it.
