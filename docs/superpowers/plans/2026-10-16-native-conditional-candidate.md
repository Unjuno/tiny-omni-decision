# Native conditional candidate implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement inline, task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one decoder-free, <=200M native conditional research candidate.
**Architecture:** Frozen VideoPrism, DeBERTa, Whisper encoder and EfficientAT,
with one small shared candidate readout. Reuse existing native preprocessing.
**Tech Stack:** Existing Python3.11/Torch2.6/Transformers5.6 and separate CPU JAX.
**Spec:** `docs/superpowers/specs/2026-10-16-native-conditional-candidate-design.md`.

## Global constraints

- Frozen existing weights/data; no training, upgrades, cloud or audit access.
- Seed17, width192, four attention heads,8192 tokens,2..40 candidates.
- Fresh outputs, original hashes/revisions, actual count<=200M; untrained status.

## Review focus

- Masked tokens must not affect scores, even if padding values change.
- Positions and features must move together except explicit temporal controls.
- Arbitrary candidate IDs must preserve probabilities under permutation.
- Exact loads reject missing keys; frozen modules remain unchanged.
- Save/reload and complete fresh native replay must support the claimed scope.

### Task1: Conditional readout

Files: `src/tiny_omni_decision/native_candidate.py`, `tests/test_native_candidate.py`.
Consumes frozen tokens and candidate-condition vectors. Produces validated token
blocks, `NativeCandidateReadout.forward(blocks, queries)` and JSON conversion.
- [x] Write/run failing tests for the spec and review-focus cases.
- [x] Implement projection, coordinates, masked attention and shared score.
- [x] Run targeted CPU tests/Ruff; commit.

### Task2: Actual native integration

Files: `scripts/probe_native_candidate.py`, config and compact result report.
Consumes pinned existing native TRAIN inputs and all four model checkpoints.
Produces input/effective config, frozen-feature artifacts, initialized head,
decisions, live parameter inventory, source hashes and terminal receipt.
- [x] Freeze paths/hashes and verify no live competing experiment.
- [x] Strict-load all encoders, reuse existing visual worker, retain full tokens.
- [x] Run raw-input inference, permutation/reuse/reload checks with no optimizer.
- [x] Run fresh-process replay to a new output and compare all feature/score arrays.
- [x] Record results and limitations; run CPU suite/Ruff/manifests, commit/push,
  confirm exact-commit CI and independent code review.

## Execution ledger

- Initial audit: HEAD32b0b2f, clean; no Python/ffmpeg processes, GPU0MiB;
  C:9.53GB free. Original full-Whisper scorer requires decoder, contrary to the
  encoder-only component budget; no existing code or artifact changed.
- Task1: 15 missing-module test failures observed, then readout/tests passed;
  committed `6fd3004` with the fixed spec/config/plan.
- Task2: input helper failures observed before implementation; native runner and
  helpers committed `6ca7adc`. Actual input path and existing dependency search
  paths were resolved before the successful smoke; no packages upgraded.
- `64736c8`: dependency provenance recorded; original smoke-01 passes in122s.
- Independent read-only review found JSON replay and staged-source gaps. Six
  regression failures preceded fixes; `4c4410a` closes both findings and adds
  whole-run RSS/provenance detail. Review confirms both fixes; no remaining
  material integration finding.
- Revised smoke-02 passes in15.79s; fresh replay-01 passes in14.69s. All21 arrays,
  probabilities and selected IDs reproduce exactly. Actual candidate196254874
  parameters; extra reload helper889345. Zero optimizer updates; no quality test.
- Full CPU suite1910passed/1skipped, Ruff pass,25manifest commands pass.
  Code commit `4c4410a4cd1cb906022868124cfba112157fb808` pushed; GitHub CI1797
  (`38781748712`) successful. Final documentation publication CI is captured in
  `native-conditional-candidate-publication-01` after push. Fresh publication
  checks also pass1910tests/1skip, Ruff and25manifest commands.
- Report builder rechecked1052current source/input pins and all output hashes.
  [Report](../../pretrained_reuse/R2_NATIVE_CONDITIONAL_CANDIDATE_20261016.md)
  records exact runs, resource limits and remaining product gates. This plan's
  functional integration is complete; the full R0–R8 goal remains active.
