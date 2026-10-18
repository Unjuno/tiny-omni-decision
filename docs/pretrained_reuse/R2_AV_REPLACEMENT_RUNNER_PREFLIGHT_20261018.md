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
the complete model output. The scorer now uses a temporary forward hook to
capture the baseline result from the original pinned model implementation. It
calculates the query-only output from that baseline and the existing correction
MLP, then records baseline/query-only/full-residual logits for each
original/self/donor input view. It reports separate donor-audio and donor-video
sensitivity summaries for baseline and query-only; the full residual's
existing sensitivity summary remains. The fixed E0G1 checkpoint's sigmoid
gates remain observable in each full-residual record. No full cross-modal
forward is repeated; one small correction-MLP evaluation is added per view.
This is a readout decomposition of one trained checkpoint, not a new model,
training run, or causal truth label.

Before retrying the runner, its required 1,115 training-time Python source
hashes were recomputed. All match after restoring the model implementation and
the preexisting pinned test fixture to their exact training-time versions.
Diagnostic-only code and tests live in new files or previously unpinned files;
the prior selected run receipt and checkpoint are unchanged.

## Verification

- The new process-snapshot test first failed because the bounded refresh helper
  did not exist, then passed with two-second reuse and refresh behavior.
- The E0G1 readout-decomposition test first failed because the three readout
  outputs were absent, then passed using actual fixture tensors. It verifies
  baseline and query-only values against direct readout calculations and checks
  that full-residual logits equal the original output.
- Scoring, AV-interaction and optional-Torch hook tests: 37 passed.
- Full CPU suite: 2,389 passed, 11 skipped.
- Ruff on changed Python files: passed.
- No GPU test or scoring attempt was started during this code change.
- The first GitHub Actions run also exposed an existing CPU-CI issue: a test
  that imports the PyTorch-dependent training runner lacked an optional-Torch
  skip. The same failure was present on parent commit `d2055f5`. The original
  training-pinned test file is unchanged. A small optional-dependency pytest
  plugin marks just that source-path test skipped when Torch is not installed;
  its behavior has a CPU-only unit test. GitHub Actions passed for the
  final source-pin-preserving code commit
  `db39767bf3a9008fa31e1387a404173c059e4dd1`.

## Live compute observation and next step

At 06:22 JST, the RTX 3080 reported an active Python CUDA PID 12648 (177 MiB),
whose Windows parent was PID 13208, the Orbitune `compound_cfe_train.py train`
process. The `idle_train.py` monitor's own-PID set contains the direct trainer
PID and supervisor PID, not descendants. Its watchdog log repeatedly reports
`foreign_gpu_process`, gracefully exits the trainer, and relaunches it. This is
consistent with the monitor treating a trainer child as foreign; no Orbitune
process or file was changed here. The outer supervisor/trainer are still live.

At the time of this preflight report, the full fixed-checkpoint diagnostic was
still unrun. That statement is superseded by the completed attempt 11 recorded
in [R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md](R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md).
Attempt 08 and the failed preflight attempts remain preserved; only attempt 11
completed all 44,375 forwards and exact self-replacement checks.

### Cooperative scheduling correction (attempts 12–13 follow-up)

Attempt 12 (`greatest-hits-postnorm-replacements-E0G1-12`) began while the GPU
was idle but stopped before its first forward after the Orbitune trainer
reacquired the GPU during CPU preflight. It took 80.2 seconds to validate the
2,755 frozen input pins; attempt 12 remains `INCOMPLETE` with zero forwards.
Attempt 13 then stopped at the initial GPU gate in 0.10 seconds after observing
Orbitune PID `6744`; it did not modify or hash the frozen inputs. The scorer now
completes the CPU integrity preflight without taking the GPU,
then waits for a clear device window before initializing its CUDA context. This
avoids aborting when Orbitune has reacquired the GPU during the lengthy CPU
checks; once a context is established, the existing monitor can observe the
diagnostic and request its normal cooperative yield. Before model transfer and
large feature-store use, and between inference batches if another GPU process
appears, the scorer waits up to 900 seconds for that process to exit. It
performs no forwards while another CUDA PID is present and fails closed on
timeout. Input/source stamp checks, periodic GPU checks, the 2-hour run limit,
and final full hash verification remain enabled.

Regression tests cover cooperative yield and timeout. The full local CPU suite
passes (2,392 passed, 11 skipped), and Ruff passes. Attempt 14 completed with
`PASS_POSTNORM_AV_REPLACEMENTS`; all 44,375 predictions and 17,750 exact
self-replacements match attempt 11 byte-for-byte in the saved prediction and
sensitivity files. Final input and source hashes passed. This scheduling change does
not alter the checkpoint, data, labels, views, replacement pairing, model
parameters, or number of forwards. Attempt 14 confirms reproducibility and is
not an independent quality estimate; attempts 12 and 13 remain preserved as
failed earlier runs.
