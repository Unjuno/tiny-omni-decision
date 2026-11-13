# RoboMNIST source-length diagnostic implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implement inline.

**Goal:** Execute the full-source fixed length-only shortcut diagnostic.
**Architecture:** Reuse pinned directory replay validation; pure grouped split,
numeric-only extraction, fixed centroid/posterior marginalization and CPU runner.
**Tech stack:** Existing Python/NumPy/decision_math, no model or dependency change.
**Spec:** `docs/pretrained_reuse/R2_ROBOMNIST_LENGTH_PROTOCOL_20261015.md`.

## Global constraints and review focus

All60 classes/2,135 groups, same-Rp across classes/views shares split; Rp01 train.
Seven lengths only; identifiers/GT excluded. Train-only standardization/centroids.
Stable unclipped NLL and full option sets; all arms retained. No source/session/
clock/OOD or Omni quality laundering.120s/2GiB CPU/16MiB outputs/1GiB disk;
legacy artifacts and8GiB GPU guard protected. No models/payloads/audit/training.

### Task1 — Tested estimator and prospective source

- [ ] Add `tests/test_robomnist_length_probe.py`; verify RED.
- [ ] Implement `scripts/robomnist_length_probe.py`: `split_blocks(blocks)`,
  `length_features(group,sizes)`, `centroid_logp(train_x,train_y,test_x,classes)`,
  `task_logp(joint_logp,task)`, `metrics(logp,targets)`.
- [ ] Verify class/view block invariance, no GT/duplicate streams, train-only
  scaling, all class coverage, finite inputs, stable NLL and exact marginals.
- [ ] Build runner `scripts/diagnose_robomnist_lengths.py`, require all70 parent
  pins and all60 lexical banks; pin complete code/libraries/protocol before fit.
- [ ] Tests/Ruff and commit before real execution in a new artifact directory.

### Task2 — Actual experiment, accounting and publication

- [ ] Execute fixed five-arm CPU diagnostic; retain config/data/predictions/results.
- [ ] Fresh-process reconstruct all inputs/splits and recompute every output/metric,
  using code/source hashes and exact native bytes, with no new fitting selection.
- [ ] Report all counts/curves (no epochs), fixed posterior tasks, bootstrap scope,
  shortcut implications and unverified clock/session/quality gates. Minimal docs.
- [ ] Full CPU/NoTorch/Ruff/manifests, read-only reviewer, normal push/exact-head CI.
