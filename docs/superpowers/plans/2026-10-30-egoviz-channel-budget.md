# EgoViz channel budget Implementation Plan

> **For agentic workers:** Execute the tasks in order in this task's existing
> isolated worktree; the user has authorized autonomous execution.

**Goal:** Measure whether a small learned channel mixer changes ranking and
calibration under a sustained, matched CPU training budget.

**Architecture:** Reuse the existing frozen feature corpus and baseline
scorer. Add a separate experimental scorer whose uniform-initialized channel
weights contribute 3,072 parameters. A new runner fits both from identical
projection initialization and sample order, selects on development NLL and
retains fixed epochs independently of that selection.

**Tech Stack:** Python 3.11, NumPy, PyTorch CPU, existing pinned feature caches.

**Spec:** `docs/pretrained_reuse/R2_EGOVIZ_CHANNEL_BUDGET_PROTOCOL_20261030.md`.

## Tasks

1. Add focused tests for initial equivalence, gate gradients, unchanged option
   permutation behavior, seed/order parity, and selection versus fixed endpoint.
2. Implement separate channel scorer and runner with hash-verified inputs,
   finite checks, frozen 512-epoch budget, per-evaluation records, checkpoint
   and per-example prediction retention. Keep existing trainer untouched.
3. Run CPU smoke in a fresh process, estimate throughput, freeze code/protocol
   in a commit, then execute the fixed comparison once in a new artifact path.
4. Reload both best and final checkpoints in a fresh process; compare saved
   scores, verify parameter counts and sample/order hashes, and record paired
   session uncertainty. Report exact curves, failures and constraints.
5. Run required checks, commit only small results, push and check the resulting
   PR CI. Keep prior artifacts and the sealed set untouched.

## Review focus

- Confirm both arms receive the identical permutation sequence and initialized
  projection weights; zero gate must reproduce the uniform mean.
- Check that gate gradients are finite and that no input encoder is trainable.
- Preserve best-NLL and fixed endpoint separately even when they coincide.
- Refuse altered inputs, nonfinite training, missing/overlapping groups, and
  output overwrite; never infer progress from an unfinished run.
- Treat development metrics and post-selection intervals as exploratory and
  distinguish a pooling change from its added 3,072 parameters.
