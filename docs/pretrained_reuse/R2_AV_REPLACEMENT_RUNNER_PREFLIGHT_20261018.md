# AV replacement diagnostic runner preflight — 2026-10-18

## Scope and prior attempt

This is an inference-only runner review for the fixed E0G1 checkpoint and the
already-frozen 8,875-question validation inventory. It does not modify any
training run, checkpoint, input feature, label, or validation selection.

Attempt 08 remains preserved at
`C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-postnorm-replacements-E0G1-08`.
It stopped as `INCOMPLETE` after 109 of 44,375 required forwards because a
foreign CUDA process appeared. It ran for 92.5 seconds, reached 116,764,160
bytes peak CUDA allocation and 1,164,251,136 bytes peak working set, and did
not produce a complete sensitivity summary. Its partial predictions are not
quality metrics.

## Runner issues found and fixed

Code inspection found that the per-view `score()` loop calls its guard before
every forward. The runner guard invoked `nvidia-smi` on every call, spawning a
process 44,375 times for one complete E0G1 pass. Attempt 08's elapsed time
includes startup and resource/hash verification, so its 0.85-second-per-forward
rate is not attributed solely to `nvidia-smi`; the subprocess placement itself
is unnecessary repeated work.

The runner now reuses its process snapshot for at most two seconds. Memory,
VRAM, elapsed-time and disk limits remain checked at every guard; input/source
hashes retain their existing 15-second cadence and final verification. A newly
visible CUDA process is detected within at most two seconds, after which the
runner fails closed. Model inputs, checkpoint, option order, replacement
inventory and number of model forwards are unchanged.

The earlier diagnostic brief also called for separating the baseline,
query-only correction and complete residual. The prior runner recorded only
the complete model output. Diagnostic mode now returns the baseline score and
the query-only correction score alongside the unchanged complete residual
score from the same model call. The scorer records baseline/query-only/full
residual logits for each original/self/donor input view and reports separate
donor-audio and donor-video sensitivity summaries for baseline and query-only.
The full residual's existing sensitivity summary remains. The fixed E0G1
checkpoint's sigmoid gates remain observable in each full-residual record.
This avoids repeating the baseline and cross-modal attention forward, but adds
one small correction-MLP evaluation per view for the query-only output. This is
a readout decomposition of one trained checkpoint, not a new model, training
run, or causal truth label.

## Verification

- The new process-snapshot test first failed because the bounded refresh helper
  did not exist, then passed with two-second reuse and refresh behavior.
- The E0G1 readout-decomposition test first failed because the three readout
  outputs were absent, then passed using actual fixture tensors. It verifies
  baseline and query-only values against direct readout calculations and checks
  that full-residual logits equal the original output.
- Scoring and AV-interaction tests: 35 passed.
- Full CPU suite: 2,388 passed, 11 skipped.
- Ruff on changed Python files: passed.
- No GPU test or scoring attempt was started during this code change.
- The first GitHub Actions run also exposed an existing CPU-CI issue: a test
  that imports the PyTorch-dependent training runner lacked an optional-Torch
  skip. The same failure was present on parent commit `d2055f5`. The test now
  explicitly skips when Torch is unavailable; local full-suite verification
  remains 2,388 passed, 11 skipped. GitHub Actions for commit
  `a5012efd5921a6c769e232cef87cd2d774735b7f` passed. The earlier failure was
  also present on parent `d2055f5` and was not caused by the AV diagnostic
  changes.

## Live compute observation and next step

At 06:22 JST, the RTX 3080 reported an active Python CUDA PID 12648 (177 MiB),
whose Windows parent was PID 13208, the Orbitune `compound_cfe_train.py train`
process. The `idle_train.py` monitor's own-PID set contains the direct trainer
PID and supervisor PID, not descendants. Its watchdog log repeatedly reports
`foreign_gpu_process`, gracefully exits the trainer, and relaunches it. This is
consistent with the monitor treating a trainer child as foreign; no Orbitune
process or file was changed here. The outer supervisor/trainer are still live.

Therefore the full fixed-checkpoint diagnostic remains unrun. Attempt 08 is
preserved unchanged. The next run must use a new output directory and the
updated committed runner, only after the GPU safety gate is genuinely clear and
the existing process handoff is within the user's authorization. It must still
complete all 44,375 forwards and exact self-replacement checks before any
sensitivity summary is interpreted.
