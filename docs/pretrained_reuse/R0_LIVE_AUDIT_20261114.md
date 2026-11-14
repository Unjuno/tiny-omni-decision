# R0 live repository and runtime audit — 2026-11-14

## Scope

Read-only refresh of the latest `docs/pretrained-reuse-roadmap-20261010`
branch (`bfb13bf6e9ac4f2ce244eb51e470afa907587380`) and the currently open
Path A / R2 pull requests. The only write was this report and its roadmap
index entry in a new worktree. No existing worktree, branch, PR, dataset,
checkpoint, feature cache, training run, process, or external artifact was
modified.

## Repository and live state

Snapshot time: `2026-11-14 15:44:14 +09:00`.

| Item | Observed state |
|---|---:|
| Open GitHub pull requests | 87 |
| Git worktrees | 158 |
| Local branches | 205 |
| `origin` branches | 196 |
| C: free bytes | 281,601,536,000 |
| Python / Pythonw / torchrun processes | 0 |
| RTX 3080 Laptop GPU | 0% utilization; 11 MiB / 16 GiB allocated |

The queried GPU snapshot reported 46 C and P8 state. The ChatGPT desktop GPU
process was listed by `nvidia-smi`; no training process used the device. No
process was stopped or restarted. This is a point-in-time snapshot, not a
promise that the machine remains idle.

The repository's main user worktree is outside this worktree. This audit used
a new branch from the latest roadmap branch and did not switch or clean any
existing checkout.

## PR and workstream reconciliation

GitHub search returned 87 open PRs. The newest five are #91–#95; each remains
Draft/Open. Current check diagnostics for #91–#95 report successful test
checks. PR #89 (Path A MiniWoB runtime) and #90 (CATER temporal evaluation)
also remain Draft/Open with successful checks. No PR was merged or retargeted.

- #91 records 1,024 fresh MiniWoB screen-only episodes. Its separate
  criterion-conditioned admission result found a deterministic RGB rule that
  solves the cohort, so this task is not suitable for measuring pretrained
  representation reuse. Do not train or reopen its sealed split.
- #92 integrates one shared MiniLM instance across current Path A routes.
- #94 adds the canonical JSONL dispatcher for Text, Audio, Image and Video.
  The fixed parity run reports 170,499,716 unique resident parameters and
  687,254,772 package bytes. Four route outputs passed JSON and saved-score
  parity; this is runtime evidence, not task-quality or fusion evidence.
- #93 keeps RoboMIND2 unadmitted because independent scenario groups and a
  typed-option task requiring both streams are not established.
- #95 keeps Localized Narratives unadmitted because image-level rights and
  voice recording reuse rights are unresolved.
- #90's CATER validation scores do not show video dependence over the
  question-only control. Its observed cohort is closed to tuning.

Other recent source screens on DROID, AIST Bimanual, TCA/FAVE, otoSpeech,
Wikimedia Commons, and commercial video candidates did not admit an M-2 task.
Existing weak or exposed evaluations were not repeated. The full current PR
list is discoverable from GitHub; this report records the latest relevant
runtime and R2 heads rather than duplicating all 87 PR bodies.

## R0–R8 gate state

- **R0:** Historical inventories and quantitative acceptance policy exist.
  Deployment device details, latency/energy ceiling, and the complete approved
  task battery are still open.
- **R1:** VideoPrism-B and other Path A checkpoints have pinned source/runtime
  evidence. V-JEPA 2.1 exact-checkpoint rights remain `REVIEW`.
- **R2:** Text, image, audio, video and screen have narrow task-specific Path A
  routes/results. No current source satisfies the rights, independent-group,
  and both-input-necessary gates for a new M-2 quality experiment.
- **R3/R4:** A shared Lens, weight merge, or alignment loss is not justified
  by current R2 evidence; do not start these experiments yet.
- **R5:** Observation reuse and basic request caching have measured evidence;
  concurrent streams and timestamped state remain open. The current JSONL
  runtime supplies a task-scoped interface, not a joint world state.
- **R6:** No compatible weight merge has been selected or validated.
- **R7:** No quantized export/reload parity or low-bit deployment runtime is
  established. The full-precision runtime package is above the provisional
  200 MB target; the target must be measured again after quantization.
- **R8:** No release candidate or locked product task battery is frozen; no
  final sealed evaluation or device handoff has occurred.

## Next action

Keep the existing Path A branches and artifacts intact. Continue only with a
bounded official-source metadata gate for a fresh task. Require per-item
commercial/ML rights, an explicit independent train/dev split with at least
200 independent groups and 1,000 examples, and a typed-option target whose
label genuinely depends on both streams. Do not download payloads or train
until these conditions pass. If no source passes, retain the negative result
and complete only product engineering gates that do not imply missing quality
evidence.
