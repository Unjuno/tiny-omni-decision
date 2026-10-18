# R0 live reconciliation — 2026-10-18

Read-only reconciliation of the current repository, open work, local process,
and the assembled Path A reference. This supplements, and does not replace,
`R0_INVENTORY.json` or `R0_LIVE_AUDIT_20261018.md`. No branch other than the
current research branch was checked out or modified; no PR was merged or closed.

## Repository and open work

- Repository: `Unjuno/tiny-omni-decision`; remote fetch completed before audit.
- Current worktree: `C:\CodexWorktrees\pretrained-reuse-omni`, branch
  `codex/textfree-video-program-order`, clean at `4d68db688bd7759c89bc65c26cc0cf919d3daaaf`.
- PR #15 remains open and draft, head
  `docs/pretrained-reuse-roadmap-20261010@43974fb2f6b070e0182465edca12d061b65ed8dd`,
  base `main@09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- Public open PR inventory is #9–#15. PRs #9–#14 are separate Teacher/QAT or
  post-quantization tracks; none was edited, merged, closed, or used as a source
  of model weights in this work.
- The repository has multiple preserved worktrees, including Teacher v1,
  video-teacher-v2-long-budget, QAT, AV, and text-free video branches. The
  read-only `git worktree list --porcelain` snapshot confirmed their paths and
  HEADs; no other worktree contents were read or changed during this check.
- The branch’s pushed result commit is covered by successful CI run
  `39118367211`. The local objective remains active; this reconciliation does
  not close R0–R8.

## Live compute and change constraints

At the 2026-10-18 19:14 JST process snapshot, the RTX 3080 Laptop GPU was using
15,821 / 16,384 MiB at 39% utilization. The compute process was PID 29208,
`compound_cfe_train.py train`, child of PID 7084 under Orbitune's watchdog and
idle-train supervisor (PIDs 26124 and 6452). The command specifies BF16,
batch 64, sequence length 512 and a yield sentinel. This is another project’s
live training; it was left running and no GPU inference, training, stop, signal,
or file change was attempted. CPU/read-only repository audits below ran without
using CUDA.

## Path A assembly revalidation

The existing, external v4 assembly was audited in place using the repository's
read-only `scripts/audit_path_a_package_size.py`, with its report written to a
new directory. No model, package file, or manifest was changed.

- Assembly: `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4`.
- Package manifest SHA-256 remains
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- All 42 manifest-listed package files passed size/hash verification.
- Reconciled unique resident parameter count: **170,499,716**.
- Current package content size excluding root manifest: **687,254,772 bytes**.
- Rechecked encoder checkpoint hashes:
  - VideoPrism-B: `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
    (114,365,184 parameters).
  - Whisper Tiny: `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`
    (37,760,640 encoder parameters in the package accounting).
  - MiniLM-L3: `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`
    (17,389,824 parameters).
- Read-only storage-only INT8 envelope: **182,018,983 bytes** at group size 256.
  This preserves every non-encoder byte and estimates 1-byte codes plus FP32
  scale and INT8 zero point per group. It created no quantized tensor, runtime
  kernel, reload test, or quality result. The actual v4 package therefore still
  exceeds the provisional 200 MB package target.
- Audit receipt:
  `C:\CodexArtifacts\pretrained-reuse-r2\current-path-a-audit-20261018\report.json`,
  SHA-256 `1c40b73e2281c608a4f1bb8aa18e805e0402788e6ab92d79a58052c3321fa4e0`.
- The audit confirmed the narrow package scope in the source report: task-scoped
  Audio, CLEVR-4 Image, synthetic Text and a CLEVRER Video scorer over already
  extracted features. This is not one end-to-end raw multimodal model; the video
  package does not include raw-video extraction, and the assembly omits the
  separate sensor Decision runtime.

## R0–R2 disposition and next work

R0 remains **partly open**: a deployment device and task-specific quality,
calibration/abstention and latency acceptance limits have not been selected.
The 100–200M parameter and <=200 MB package values remain engineering targets.

R1 has a usable, rights-cleared VideoPrism-B reference with pinned weights,
source, and real image/video forwards. V-JEPA 2.1's exact checkpoint grant
remains `REVIEW`; it is not needed to use the VideoPrism Path A reference.
Existing model, source, and rights findings are retained in
`R1_AUDIT.md`, `R1_RELEASE_LICENSE_FOLLOWUP_20261011.md`, and
`R1_CANDIDATE_INVENTORY.json`.

R2 remains open. Existing readouts show task-specific capability, not a
95–98% Omni model: text-free Audio is 96.56% Accuracy but materially worse
calibration than its comparator; CLEVR-4 Image macro Accuracy is 88.30% with
count at 58.19%; selected CLEVRER typed-program order probe favored the mean
baseline's Accuracy/selector and both arms overfit; HARTH sensor results are
below 95% Accuracy and have only three validation subjects; the near-command
audio/video correspondence task is 84.08% on validation. These observed
development splits must not be reused for checkpoint selection.

The next quality work should therefore establish a new, rights-cleared,
independently grouped task/split for the weakest task or a both-required joint
decision, then compare a frozen Path A baseline against the matched control.
Do not add Path B/C, quantization, or another repeated fit on these same
validation IDs until a concrete Path A inefficiency and a valid new evaluation
set justify it. The present GPU occupancy makes a new GPU run unavailable
without affecting the live Orbitune job; CPU audit work can continue.
