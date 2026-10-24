# Shared native readout gradient diagnostic protocol

Status: frozen before execution. This is a TRAIN-only descriptive diagnostic,
not model selection, a quality evaluation, or evidence of product readiness.

## Question

At the selected checkpoint of the completed repeat-coverage fit, do the four
modality-specific losses tend to push the shared readout in aligned or opposing
directions, and how often would their combined gradient trigger the fit's
global norm-1 clipping rule?

## Frozen references

- Repository source commit: `b3b7bb3abc4d27b6da51aa05032d2ff14704362f`.
- Run: `C:/CodexArtifacts/pretrained-reuse-r2/native-coverage-repeat-fit-01`.
- Checkpoint: `best.pt`, selected by the already-frozen validation macro-NLL
  rule at optimizer step 2,048. The run reports `best_step=2048`; its bytes
  must match `checkpoint-2048.pt` (SHA-256
  `94d4e452314d0f41f43c4f1711711837c38b60e9eea7816c2149ad3636d888f2`).
  Validation is not loaded by this diagnostic.
- Data: the exact repeat-run sampler receipt and its original TRAIN split in
  `native-common-corpus-02`. Only `train.jsonl`, `observations.json`, `banks.json`
  and the referenced frozen feature banks/query cache are read. The validation
  file and sealed audit are never opened.
- Optimizer history, labels or weights are not changed. No fitting, calibration,
  new seed, checkpoint selection, or architecture intervention occurs.

The run receipt's `source_commit` is `b3b7bb3`, but five validated runtime
dependency files do not match those files at that commit. Their exact
`validated_hashes` bytes are bundled under
`docs/pretrained_reuse/source_snapshots/native_coverage_repeat_fit_01/` and
loaded explicitly by the audit. The historical commit that contained all five
dependency bytes is unknown. This provenance mismatch is retained rather than
described as a fully commit-reproducible run.

## Fixed measurement sample

Use the exact `order` and `occurrence_plan.occurrences` stored by the 2,048-step
repeat run (sampler SHA-256
`23b049b25495dc2d7a8e983b1abb3b257399413c975b2901ed8c7af171073797`).
The selected row IDs and occurrence order are frozen in
`configs/pretrained_reuse/native_shared_gradient_audit_v1.json`; their row ID
order SHA-256 is
`82f4e8ff288b4e29faf52c35b8c15c7a9e9c51a3cfbbd1bbd7af06b0cbdad11e`.
The frozen manifest SHA-256 is
`fc9dda42df0a683cbbb1bdb4e834137eba885491419802543e995974adb17cf7`.
Select 32 optimizer updates with zero-based indices
`floor(i * 2047 / 31)` for `i=0..31`, covering optimizer steps 1 through 2,048.
Each recorded update contains eight examples, exactly two per modality, as in
the original sampler. Preserve the saved occurrence IDs and reproduce the
original option permutation function. The selected 256 presentations contain
253 unique rows and 3 repeated presentations in the frozen sampler selection.
These repeats are kept and counted; no resampling occurs.

For every selected update, compute the mean cross-entropy gradient separately
for each modality across its two examples. Report all six pairwise cosine
similarities, each modality gradient norm, the cancellation ratio
`||sum(g_m)|| / sum(||g_m||)`, and each modality's projection share on the
combined direction. The original optimizer averages eight individual losses,
so its unclipped batch gradient is `sum(g_m) / 4`; derive the norm-1 clipping
factor and clip frequency from that exact scale. Do not update parameters.

## Interpretation rule

This audit is descriptive only. Report per-pair mean, median, sample standard
deviation, range, and negative-cosine fraction over the 32 fixed observations.
Opposing directions are evidence of local gradient conflict on these specific
TRAIN batches, not proof of persistent interference. Aligned gradients do not
prove that the shared representation is sufficient. The audit cannot choose an
architecture or optimizer by itself; combine it with the existing paired
coverage results and later task-specific transfer evidence.

## Runtime and integrity

Use the local RTX 3080 Laptop only through the cooperative HotUGPU reservation
guard (`reserve 3200 MiB`, block below 14,000 MiB free, require at least 8,000
MiB free to the child). If Orbitune does not yield, the child must not start.
No cloud compute or paid resources. Verify the selected checkpoint, run receipt,
sampler, corpus metadata, selected feature banks and all measurement IDs. Record
the effective source hash, GPU/runtime, reservation, start/end time, elapsed
time, peak VRAM, and exact outputs under a new artifact directory. Existing
runs/checkpoints remain read-only.

## Predeclared outputs

- `preflight.json`: checkpoint/data/sampler hashes and the exact 256 TRAIN IDs,
  source/task/modality counts, observation and group counts, and HotUGPU receipt.
- `measurements.jsonl`: 32 per-update gradient measurements and sampled IDs.
- `result.json`: integrity status, environment, resource use, aggregate
  gradient statistics, and hashes for the two outputs above.

An execution failure is reported as blocked/incomplete and does not imply that
the learning-quantity or model-capacity hypothesis failed.
