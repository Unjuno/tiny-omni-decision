# Reproducibility isolation after numeric128-prefix failure

Two bounded diagnostics, zero optimizer updates, no validation or audit inference:

1. Selected embedding backward with actual first training batch token IDs and
   fixed synthetic upstream gradients:30 repetitions with deterministic algorithms
   off and30 on. All gradient hashes identical within/between these modes.
2. Actual adapted Cross-Encoder + numeric delta with real first batch loss:
   five forward/backward repetitions per mode. All forward hashes identical;
   no trainable parameter gradient varied within either mode.

Input[25,512], RTX3080Laptop, Torch2.6.0+cu124, SDPA in full model.
Both diagnostic processes set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA
initialization. The old training did not record that variable. Therefore these
checks do NOT reproduce all historical runtime conditions or rule out numerical
variation across processes, later minibatch shapes, optimizer updates or allocator
state. No claim that deterministic=True alone fixes the failed2048 run.

The simple hypothesis of unstable repeated embedding backward on this batch was
not reproduced. Exact cause remains UNKNOWN. Do not change the quality selector,
relax the exact-prefix gate or repeatedly launch long runs hoping for a match.

Runner instrumentation now records runtime determinism/TF32/SDPA/cuBLAS fields
and, on future prefix failure, the actual and expected records plus differing
fields before re-raising. It does not alter computation or permit continuation.
The old stopped run cannot retroactively supply its missing history record.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-embedding-determinism-01`
and `numeric-full-backward-determinism-01`. No checkpoint or training data changed.
All scripts and measured results retained;399CPUtests pass,1skip; Ruff passes.
2048-step comparison remains unexecuted beyond128, not a quality failure at2048.
