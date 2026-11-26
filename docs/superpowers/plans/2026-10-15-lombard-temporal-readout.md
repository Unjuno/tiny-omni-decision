# Native temporal command readout implementation plan

**Spec:** `docs/pretrained_reuse/R2_LOMBARD_TEMPORAL_PROTOCOL_20261015.md`.
**Goal:** Compare native chronological visual features against broadcast mean
under identical readout capacity, data, initialization and optimizer budget.
**Execution:** Inline under the user's autonomous local-research authorization.

- [ ] RED/GREEN exact bank/time interventions and fixed106,355-parameter head,
  explicit missingness, CPU gradients/order sensitivity/Save-Reload tests.
- [ ] Extend independent source/cache loader with whole16-frame SHA/mean/ID
  checks, full versus four-train-only scope and immutable dependency pins.
- [ ] Implement five-arm CPU-only trainer: fixed24epochs, histories/predictions,
  independent validation selector/calibration, best/final/optimizer/RNG states,
  source rechecks, safe bounds, full resource/sampling counts and JSON parity.
- [ ] Commit protocol/code before separate-process training-only smoke; admit
  full run only after measured gradient/finite/reload/resources/throughput gates.
- [ ] Actually run all five arms, fixed selected controls, grouped intervals and
  independent prediction/report checks; fresh best/final checkpoint reload.
  Full execution uses a separately reviewed controller/evaluator; this first
  trainer CLI admits smoke only and its executed source pins remain immutable.
- [ ] Publish all outcomes without legacy changes or general Omni promotion;
  Ruff/CPU/no-Torch/manifests/commit/push/exact-head CI.

All4080/398/889 native grouped examples and51options remain unchanged. No
encoder training, signal noise, extra frames/seed/budget, cloud or sealed audit.
CPU4threads/FP32/deterministic,7200s/3GiB peak/128MiB new output/1GiB disk reserve;
existing GPU8GiB guard and interrupted runs stay unchanged.
