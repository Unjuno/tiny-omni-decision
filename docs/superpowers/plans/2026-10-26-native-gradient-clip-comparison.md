# Native Decision gradient clipping comparison plan

**Goal:** Measure one fresh 32,768-update Path A run with global gradient clip 2.0 against the preserved clip-1.0 run.

**Architecture:** Keep every existing model, feature, sampler and validation policy fixed. Add a bounded CLI override for clip norm and validate GPU ownership through the cooperative HotUGPU gate.

**Tech Stack:** Existing Python 3.11 / PyTorch 2.6 CUDA environment, pytest, Ruff, local RTX 3080 Laptop.

**Spec:** `docs/superpowers/specs/2026-10-26-native-gradient-clip-comparison.md`.

## Global Constraints

- Only clip norm changes: 1.0 baseline to 2.0 candidate.
- Use seed 17, 32,768 updates, 8 microbatches/update, unchanged repeat pool and full-horizon cosine schedule.
- Preserve all existing artifacts and use fresh output directories.
- Do not load calibration, transfer, test or sealed-audit data.
- Use only local compute and the cooperative HotUGPU reservation guard.

## Review Focus

- A partial or stale HotUGPU environment must not bypass the idle-GPU gate.
- The static baseline config must remain byte-identical while effective config records clip 2.0.
- Smoke/train/replay references must reject mismatched clip thresholds.
- Candidate output must preserve best and final checkpoints separately.
- Existing clip-1.0 execution must remain unchanged by default.

---

### Task 1: Configurable clip threshold and cooperative GPU admission

**Files:**
- Modify: `scripts/train_native_long_budget.py`
- Modify: `scripts/native_long_budget.py`
- Test: `tests/test_native_long_budget.py`

**Interfaces:**
- Add a pure helper that copies the pinned baseline config and changes only `training.clip_norm` for the allowed values 1.0 and 2.0.
- The runner CLI accepts `--clip-norm 1.0|2.0`; default remains 1.0.
- Use `validate_gpu_preflight` to accept either an idle GPU or a valid cooperative reservation.

- [x] Add tests for clip-config equality outside the cap, invalid caps, and preserved default behavior.
- [x] Run tests and verify the new tests fail before implementation.
- [x] Implement CLI/effective-config/provenance/reference checks, dynamic clip call and fail-closed GPU preflight.
- [x] Run targeted tests, Ruff and the full CPU suite (2,636 passed, 11 skipped); commit/push completed in `533c3c5`.

### Task 2: Execute and compare the fixed candidate

**Files:**
- Create: `docs/pretrained_reuse/R2_NATIVE_CLIP_NORM_RESULTS_20261026.md`
- Modify: `ROADMAP.md`
- Artifacts: fresh directories outside Git under `C:/CodexArtifacts/pretrained-reuse-r2/`

**Interfaces:** Consume the pinned norm-1.0 fit and the new runner's norm-2.0 smoke/train/replay receipts.

- [x] Run a 32-update smoke with clip norm 2.0 under HotUGPU reservation.
- [x] Verify effective config, learning-rate/sample-order identity and resource projection before the main fit.
- [x] Complete the 32,768-update candidate run without performance early stopping; preserve best/final checkpoints.
- [x] Fresh-process replay best/final and compute the predeclared comparisons to clip 1.0.
- [x] Record metrics, compute, limitations and exact hashes in the R2 report.
- [x] Run Ruff, the full CPU suite, manifest/catalog validation and diff checks.
- [ ] Commit/push the report and roadmap update; verify target CI.

