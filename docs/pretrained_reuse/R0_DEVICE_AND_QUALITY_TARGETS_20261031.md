# R0 engineering device and quality targets — reconciled 2026-10-31

This note brings the user-confirmed engineering criteria from commit
`87265cf2c59608b806e4975c17f9a143dbbfe0ab` on
`codex/r2-audio-vibration-source-20261027` into the current roadmap branch.
It records engineering assumptions only; it does not declare a benchmark or
product release gate passed.

## Confirmed for engineering

- Reference device: local Windows PC with NVIDIA RTX 3080 Laptop GPU (16 GiB);
  CPU is the bounded fallback. No paid or cloud compute.
- Model budget: 100–200M total resident parameters before quantization.
- Package target: at most 200 MB, counted only after a real quantized export,
  save/reload parity check, and complete package accounting.
- Accuracy aspiration: at least 95% on each separately defined task, with 98%
  as a stretch target. Unrelated tasks must not be pooled into a pass.
- Report task-specific Accuracy, NLL, Brier, ECE, reliability bins, independent
  groups, confidence intervals, wall time, GPU use and peak VRAM.
- Sensor input remains optional and outside the first acceptance suite until
  an approved sensor task is available.

## Still open

- No numeric NLL, Brier, ECE or reliability acceptance floor is approved.
- No numeric wall-clock ceiling or release latency target is approved.
- These engineering targets do not grant data/model rights, close R2's
  both-input-required task gate, or qualify any model for release.

