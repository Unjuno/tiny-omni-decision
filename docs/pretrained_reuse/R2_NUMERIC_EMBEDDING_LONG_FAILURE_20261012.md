# Numeric2048 run: stopped at exact-prefix gate

Execution INCOMPLETE. Source5c51cc7, runtime session87106 returned exit1 at128
updates after initial step0 matched. No live training process remains. The
predeclared exact512-prefix gate stopped the run before further training; it was
not weakened. Old512 run and both baselines remain untouched.

Saved128-step predictions have identical IDs/options/targets. Compared with the
old512 prefix, maximum probability differences are2.2054e-6(original) and
3.5167e-6(curriculum). No argmax changes in either source. These small differences
still violate the declared exact-reproduction gate. The current log does not
identify the first differing history field, so no specific CUDA kernel or code
change is asserted as the root cause. Source/order/initial metric agreement is
insufficient evidence of bitwise-equivalent training.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-embedding-long-01` has
config, step0 history, step0 best weights, step128 predictions and failure analysis.
No completed result or final checkpoint exists. Step128 optimizer/model/RNG state
was not saved; this cannot be reported as exact-resumable. A future retry would
need fresh initialization in a separate directory, not a silent resume.

Do not count this as evidence that numeric updates fail at2048. Neither2048
comparison nor final model selection has been executed. Before retrying, improve
failure diagnostics and establish an explicit reproducibility protocol; do not
repeatedly restart until numerical coincidence passes. No new quality artifact.
CPUtests399pass,1skip; branch CI green before this failure report. Full goal unmet.
