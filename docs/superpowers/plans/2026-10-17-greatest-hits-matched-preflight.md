# Greatest Hits matched preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the full matched TRAIN sampling plan and verify shared fitted
initialization before admitting the full AV learning comparison.

**Architecture:** A pure deterministic five-stream sampler and validation-only
retention selector feed the future trainer. A separate read-only preflight writes
the real full plan, counts and initial model states into a fresh local directory.
It does not read the active partial video cache or run model training.

**Tech Stack:** Existing Python3.11/PyTorch2.6 environment, local filesystem.

**Spec:** `docs/pretrained_reuse/R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md`.

## Global Constraints

- Seed17;32,768 updates;4 joint +1 each Text/Image/Audio/Video microbatch.
- Fresh optimizer after exact common step8192 model warm start; total<=200M.
- No repeat within a stream until all its rows are consumed; labels never sample.
- Preserve all old files and live video producer. No audit/calibration/test access.
- Full training remains gated on complete caches and measured full-path cost.

## Review Focus

- Unequal bucket exhaustion must skip empty queues without premature repetition.
- Input row order must not change the sequence; duplicate IDs/validation rows fail.
- Label corruption must fail validation, never silently alter an option mapping.
- Macro improvement must not override a failing individual-modality retention gate.
- Warm start must copy all baseline tensors and preserve zero output correction.

### Task 1: Tested matched sampler and selector

**Files:** Create `scripts/greatest_hits_matched.py`,
`tests/test_greatest_hits_matched.py`.

**Interfaces:** `build_plan(joint, common, steps=32768)` returns an ID batch plan,
per-stream cycle/exhaustion evidence and stable SHA. `presentation(row, occurrence)`
returns full candidate order and aligned target. `select_checkpoint(best, joint_nll,
retention, initial)` returns eligibility, selection and explicit failure reasons.
Reuses the existing long LR schedule unchanged.

- [x] Write failing tests for determinism, exhausted buckets, exact8-item batches,
  full coverage before repeats, invalid IDs/splits/options/targets, label-blind
  sampling, full option alignment and retention/selection/tie behavior.
- [x] Run `python -m pytest tests/test_greatest_hits_matched.py -q`; expect RED.
- [x] Implement those interfaces and rerun; expect all tests PASS.
- [x] Commit the tested sampler/selector and protocol.

### Task 2: Real full-plan and warm-start evidence

**Files:** Create `scripts/preflight_greatest_hits_matched.py`,
`scripts/greatest_hits_initialization.py`, `tests/test_greatest_hits_initialization.py`.

**Interfaces:** `initialized_model(arm, control_state, fixture_state)` returns a
strictly loaded native model with exact copied baseline and unchanged residual.
Preflight consumes Task1's plan/selection contracts and exact pinned real TRAIN
corpora/control/fixture receipts. It emits plan, prefix/class counts, common initial
states, hashes and a terminal receipt; it refuses an existing output directory.

- [x] Write and run RED initialization tests for all five arms, baseline parity,
  disabled policies, malformed/nonfinite control, nonzero correction and budget.
- [x] Implement initialization and preflight; run tests GREEN, full CPU suite,
  Ruff and existing manifest commands. Expected: no failures.
- [x] Commit execution source, run actual full metadata preflight into new output,
  independently check counts/IDs/states and saved hashes. Expected: deterministic
  full plan, valid frozen source linkage, retained states, no optimizer updates.
- [x] Fresh whole-change review: no actionable material findings in fe79994..4f90bbd.
- [ ] Publish measured evidence and confirm exact pushed-commit CI in the external
  verification receipt. Do not claim full training completed.

## Execution ledger

User's standing autonomous execution authorization applies; native execution is
used. A small fresh review at completion follows the executing-plans skill.
The original AV spec's broad both-required gate remains unmet; the scoped native
field comparison and this preflight do not satisfy that product requirement.
