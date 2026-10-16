# Matched AV runner implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute the frozen matched AV protocol through a guarded local runner,
with complete evaluation, separate best/final saves and reload evidence.

**Architecture:** Reuse the passing sampler and initial states. A separate input
audit admits complete immutable feature banks; one shared runtime implements
the exact eight-example update and evaluation for all five arms. Separate smoke,
train and replay processes preserve initialization and log every outcome.

**Tech Stack:** Existing Python3.11/PyTorch/NumPy environment; no upgrades.

**Spec:** docs/pretrained_reuse/R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md

## Global Constraints

- Preserve all old artifacts and the live video producer; never use its partial cache.
- Seed17,32768 updates,4 joint+1 of each original modality,unchanged options/CE/LR.
- Encoders frozen; total197097949 maximum; no sealed/calibration/test payloads.
- Full fixed validation and unique consumed TRAIN; retention-gated joint NLL selection.
- Separate fresh outputs; no early performance stop, no automatic resume claim.
- Actual full-path cost measurement precedes admission; resource failure is incomplete.

## Review Focus

- Common query permutation must match joint permutation and remapped target.
- Whole consumed TRAIN union is fixed across curve points, never a rolling loss gap.
- Failed/partial caches and changed sources must fail before optimizer updates.
- A late ineligible checkpoint cannot overwrite best; final and reload remain distinct.
- Raw score matrices must retain exact IDs/options/targets and missing-view meaning.

### Task 1: Shared matched runtime

**Files:** Create scripts/greatest_hits_matched_runtime.py and tests/test_greatest_hits_matched_runtime.py.

**Interfaces:** Consume presentation/select_checkpoint and existing stores/metrics.
Produce metric_rows, MatchedStore.get, update, evaluate, checkpoint helpers.

- [ ] Write tests for exact eight-loss gradient, both query permutations, full probe
  membership, target metadata, nonfinite rejection, gate capture and checkpoint policy.
- [ ] Run tests and observe missing implementation failures (RED).
- [ ] Implement minimal shared runtime without changing historical trainers.
- [ ] Run tests (GREEN), record evidence and commit.

### Task 2: Audit and CLI execution

**Files:** Create scripts/greatest_hits_matched_inputs.py,
scripts/train_greatest_hits_matched.py and tests/test_greatest_hits_matched_runner.py.

**Interfaces:** Consume Task1 runtime and frozen preflight payloads. Produce audit
receipt, per-arm smoke/train/replay results, raw scores, history and checkpoints.

- [ ] Test closed admission, exact phase/arm/config contracts, retention best/final
  bookkeeping and source/input drift before execution; observe RED.
- [ ] Implement full bank/clock/smoke parity audit and guarded phases. Each smoke
  uses32 exact-plan updates and full evaluation; project32768 steps and ten full
  evaluations. Train requires passing same-source/data/arm smoke and a recorded
  resource allowance, with no scientific budget overrides. Replay re-evaluates
  selected and final saves against saved full matrices.
- [ ] Run targeted then full Ruff/pytest/manifest verification. Review whole change.
- [ ] Commit/push code and inspect exact-head CI. Run full audit and smoke only
  after producer completion; report unexecuted phases truthfully if still running.

## Execution decisions

The existing prospective scientific protocol is unchanged. The CLI runs one arm
per process, sequentially, so failure cannot silently proceed to another arm.
Full evaluation includes joint TRAIN, common TRAIN and all three joint validation
views plus common validation. TRAIN probe means the full planned consumed union
(including at step0), with actual cumulative exposure separately recorded.
