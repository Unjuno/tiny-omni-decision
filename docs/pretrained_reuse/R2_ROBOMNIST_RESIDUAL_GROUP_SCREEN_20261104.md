# RoboMNIST residual-group eligibility screen — 2026-11-04

## Decision

The existing RoboMNIST AV-match experiment has no sufficiently large residual group pool for a clean follow-up evaluation. Do not re-evaluate its exposed validation set, fit against the ten residual groups, or treat the other 345 source groups as unused. Keep the existing train-only checkpoint and completed feature cache unchanged.

## Evidence inspected

- Pinned source split: `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-split-20261103-attempt-03\manifest.json`, SHA-256 `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Split implementation in repository commit `bbec1cfb317f7d94b2fb346f4b89d0527ecc290c`: `scripts/robomnist_av_match_split.py`, SHA-256 `aa6d6e9cadac345053d54c31130bf34dab8d7414d51ed6face8f61b5024dce41`.
- The pinned split code builds its eligible pool only from a prior non-audit run's summary, train and validation manifests, verifies their hashes, and fails closed unless payload, training, model inference and sealed-audit evaluation were all recorded false.
- The resulting manifest records 2,135 complete source repetitions, a prior non-sealed eligible pool of 1,790, 780 train groups, 1,000 validation groups, 1,780 selected groups, and 10 prior non-sealed groups not selected. The total 355 source groups outside the selected train/validation set is therefore not a clean residual pool: only ten are explicitly marked as unselected from the eligible prior non-sealed pool; the other 345 are outside that pool and were not inspected.

## Interpretation

The task feasibility report's provisional validation gate is 1,000 independent groups. Ten residual eligible groups cannot meet it and do not justify feature extraction or a new model run. The existing full validation has already been evaluated once and showed only a modest accuracy lift with severe calibration failure; it remains closed to further tuning. The omitted 345 groups may include protected or previously exposed records, so their availability and status remain unknown by design.

This screen used split-level counts and implementation logic to decide eligibility. It did not run inference, recompute metrics, use the ten residual groups, inspect the omitted 345 group identities or payloads, or access sealed/test data. Existing model, feature-cache, split, and prediction artifacts were left unchanged.

## Next gate

Seek a separate rights-cleared source/task with a sufficiently large, independently grouped train/calibration/development partition and a fixed Decision target. If none is available, keep R2-M open and do not manufacture additional examples or relabel the current development validation as blind.
