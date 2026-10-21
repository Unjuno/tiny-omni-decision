# Fixed gated fusion run

Freeze before execution: one seed17, CPU float32, same VIDIMU130train/39validation
trials and cached features as the block-scale run. Do not load audit. Use the
same train-only per-coordinate normalization and sqrt(360/6144) video scaling,
full-batch AdamW lr0.001/weight_decay0.01, 100epochs. Evaluate epochs0..100 and
select the earliest minimum validation NLL. Keep all predictions, best/final
weights, config/hashes and curves. No new seeds, ratios or epoch extensions.

Change the readout architecture only: named video6144/sensor360 projections to
width64, masked learned mixture, shared option-conditioned MLP. Options are
13 fixed one-hot activity descriptors. The two existing placement indicators
enter the optional context/criterion vector (always present); they are not a
user instruction or language feature. All classes, placement confound and task
limitations remain unchanged. Count all trainable parameters and frozen
VideoPrism114365184. This run bundles projection, nonlinear fusion and shared
candidate scoring; it cannot attribute a difference to gating alone.

Compare selected metrics and subject-cluster paired NLL differences against
sensor-only, video-only and block-scaled linear fusion. Same previously fixed
within-placement replacement mapping for video/sensor shuffles. Also record
explicit absent-video/absent-sensor masks, which are out of the training
distribution; do not equate them with prior linear zero-feature interventions.
Verify candidate permutation and exact checkpoint reload. Report gradients,
wall time, and complete curves; full process memory is measured separately from
parameter bytes if available. No encoder updates or GPU execution.

Architecture acceptance requires improving on both singletons, not simply on
the prior joint arm. This is an already-observed, three-subject development
validation; no generalization, independent-test, temporal synchronization or
95% product claim. Stop this run after its fixed budget regardless of result.
