# Sensor subject diagnostics: transitions remain weak

Read-only analysis of the already selected checkpoints; no training, new
checkpoint selection, calibration or sealed-audit access. Exact six-channel
float32 physical-value window hashes have **zero train/validation overlap**.
This does not establish absence of near duplicates; overlapping windows within
each subject remain correlated. Prepared subject hashes were revalidated.

| Subset | Windows | Temporal Accuracy / NLL | Static Accuracy / NLL |
|---|---:|---:|---:|
| Mixed activity windows |1,178|0.515280 /1.162224|0.589134 /1.050663|
| Unmixed windows |11,585|0.955287 /0.147586|0.974277 /0.101783|

The target is the center-sample label. Both models pool the entire window; activity
transitions are substantially harder. The high unmixed score must not replace
full-set quality: removing transitions would simplify the benchmark after seeing
results. Retain all windows. This suggests testing a center-aware readout as one
controlled next hypothesis, not claiming that transition ambiguity is the only
cause. The temporal model's higher balanced Accuracy also requires retaining
rare-class analysis rather than selecting only sample-weighted winners.

Exhaustive paired subject-cluster bootstrap enumerates all27 resamples of3 subjects
with replacement. Linear percentile95% intervals for temporal minus static:
Accuracy[-0.046836,-0.000620], NLL[0.003052,0.090628],
Brier[0.002159,0.058010]. These discrete intervals are descriptive and fragile
with only3 independent subjects and validation-based model selection. No blind
generalization claim follows from their excluding zero.

Per-subject/per-class metrics and confusion matrices are retained in the
[complete diagnostic](R2M_SENSOR_SUBJECT_DIAGNOSTICS_20261012.json), with class
indices mapped to activity codes in the original run config. Previous checkpoints
and predictions remain unchanged. No architecture or precision change adopted.
R2-M sensor fusion, screen path, free custom semantics and Omni95–98% remain unmet.
