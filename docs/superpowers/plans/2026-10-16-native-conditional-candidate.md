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
- [ ] Write/run failing tests for the spec and review-focus cases.
- [ ] Implement projection, coordinates, masked attention and shared score.
- [ ] Run targeted CPU tests/Ruff; commit.

### Task2: Actual native integration

Files: `scripts/probe_native_candidate.py`, config and compact result report.
Consumes pinned existing native TRAIN inputs and all four model checkpoints.
Produces input/effective config, frozen-feature artifacts, initialized head,
decisions, live parameter inventory, source hashes and terminal receipt.
- [ ] Freeze paths/hashes and verify no live competing experiment.
- [ ] Strict-load all encoders, reuse existing visual worker, retain full tokens.
- [ ] Run raw-input inference, permutation/reuse/reload checks with no optimizer.
- [ ] Run fresh-process replay to a new output and compare all feature/score arrays.
- [ ] Record results and limitations; run CPU suite/Ruff/manifests, commit/push,
  confirm exact-commit CI and independent code review.

## Execution ledger

- Initial audit: HEAD32b0b2f, clean; no Python/ffmpeg processes, GPU0MiB;
  C:9.53GB free. Original full-Whisper scorer requires decoder, contrary to the
  encoder-only component budget; no existing code or artifact changed.
