# Provisional quantitative acceptance policy — 2026-10-31

Status: internal engineering policy for future, newly frozen candidate
evaluations. It is not a user-approved product release contract and does not
retroactively change the interpretation of any previously observed result.

## Evidence and accuracy

- Evaluate each named task separately. Do not pool modalities or tasks.
- A task is eligible for a PASS decision only with at least 1,000 examples and
  200 independent source groups. Below either threshold, report the estimate
  and interval as low-power and classify the gate `UNCERTAIN`.
- The provisional accuracy target is 95% point accuracy and a lower endpoint
  of at least 95% for a two-sided 95% source-group bootstrap interval. 98% is a
  stretch target, not the pass threshold.
- Use 10,000 paired source-group bootstrap replicates with seed 17, as defined
  in `EVAL_PROTOCOL.md`. Preserve group membership and report the effective
  number of independent groups.

## Calibration and proper scoring rules

- Report NLL, multiclass Brier, ECE-15, reliability bins, and mean confidence
  for every task.
- Provisional calibration gate: ECE-15 point estimate <= 0.05 and upper
  two-sided 95% source-group bootstrap endpoint <= 0.05.
- Provisional proper-score gate: candidate NLL and Brier must each improve on
  both uniform-option and train-only majority-class baselines; the upper
  endpoint of paired source-group bootstrap differences
  (candidate minus baseline) must be below zero.
- Any temperature or other calibrator must be fitted only on a separately
  identified training-group calibration partition. Never fit it on validation
  or audit groups.

## Abstention and joint-input tasks

- Count abstention as incorrect in all-query accuracy. Also report covered-only
  performance and coverage. Provisional minimum coverage is 95%, with a lower
  95% source-group bootstrap endpoint of at least 95%.
- For a task claimed to require multiple inputs, freeze matched unimodal,
  missing-input, shuffled-input, and stale/time-offset controls before fitting.
  The fused candidate must improve both accuracy and NLL over the best
  unimodal control under paired group-level uncertainty; otherwise the
  both-input-required claim remains unresolved.
- Report missing, stale, and out-of-distribution controls separately.

## Provisional local resource budget

These are planning caps for a new candidate experiment, not claims about
existing runs or a deployment latency promise:

- Plumbing smoke: <=15 minutes and <=128 examples.
- Frozen-feature extraction: <=4 GPU-hours per candidate.
- Readout fitting plus scheduled validation: <=4 GPU-hours per candidate.
- One locked evaluation: <=2 GPU-hours.
- Total: <=10 GPU-hours per candidate, local hardware only, $0 cloud spend.
- Before an expensive run, perform a >=10-minute throughput dry run where
  practical. Record examples/second, wall time, peak VRAM, output bytes and
  the stop time. Keep at least 2 GiB free on the artifact volume.
- Do not claim exact resume unless optimizer, scheduler, sampler position,
  consumed IDs and RNG state are all restored and verified.

The eventual end-user device, task mix, latency ceiling, energy target and
complete approved task battery remain open. These provisional caps cannot
close R0 or qualify a model for release. Existing validation/audit results
remain as originally observed; none may be reclassified as blind.
