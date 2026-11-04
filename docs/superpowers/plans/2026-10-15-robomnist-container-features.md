# Native container invariance and frozen features implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implement inline.

**Goal:** Execute/replay actual decoder and frozen pretrained visual features.
**Architecture:** New pure append/whole-lens/digest helpers; bounded CPU runner;
source/control artifacts and independent fresh-model saved-product replay.
**Tech stack:** Existing Python/NumPy/Pillow/PyAV/PyTorch/Transformers, no upgrades.
**Spec:** `docs/pretrained_reuse/R2_ROBOMNIST_CONTAINER_FEATURE_PROTOCOL_20261015.md`.

## Global constraints/review focus

Fixed known-train pilot/all3camera; exactly1MiBfree atom per new derivative.
All450RGB/PTS per camera unchanged; fixed3frames×2wholelenses/18vectors.
TinyCLIP83,101,185fullyresident/frozenCPUfloat32/2threads/seed17/eager;
original/derivative/uniform fixed controls, no temporal averaging or quality claim.
600s/3GiB CPU/128MiB outputs/1GiB disk; all legacy artifacts/8GiB GPU guard intact.
No output overwrite, source-label inputs, silently cropped lenses, fabricated
monitoring, late hash binding, partial/reload-omitted success or source-CRC misuse.

### Task1 — Tests and actual producer

- [ ] RED→GREEN tests `tests/test_robomnist_container_features.py` for exact atom
  layout/source preservation/no overwrite, stereo split/aspect/padding and
  malformed shape/dtype, full array digest/equality and unavailable monitor faults.
- [ ] Implement `scripts/robomnist_container_features.py`: `append_free_atom`,
  `stereo_tiles`, `digest_arrays`, `same_arrays`; optional dependencies lazy.
- [ ] Implement `scripts/probe_robomnist_container_features.py`: bind all raw/
  diagnostic/released-model/library/protocol pins, decode derivatives, fixed
  length control and three actual frozen forward conditions; canonical outputs.
- [ ] Tests/Ruff/pre-execution commit; execute one new bounded output, observe
  terminal state and preserve failures without changing the profile.

### Task2 — Fresh reload and publication

- [ ] Add saved-product/fresh-model replay and corruption regression tests.
- [ ] Separate process reopens all derivatives and reloads exact model; recompute
  every pixel/feature/posterior/state/control and terminal evidence binding.
- [ ] Small results/measurements/limitations and minimal README/ROADMAP notes;
  full CPU/NoTorch/Ruff/manifests, read-only reviewer, normal push/exact-head CI.
