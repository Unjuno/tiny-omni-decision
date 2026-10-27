# DROID train-recording action-change coverage

Freeze before reading the original shards. Use exactly the prior fixed 275
training recording IDs and the previously pinned receipt and parser. Verify
all original shard lengths and SHA-256s; parse identity of nonselected records
and skip their numeric data. For selected records, verify recording identity,
source length and the existing fixed target label. Do not decode images or
video; no validation, test or audit labels are selected.

Using the existing dominant Cartesian command rule and fixed threshold 1e-4,
count adjacent `(command[t-1], command[t])` pairs for `t >= 8` until the end of
each selected train recording. Report total transition count, same/change counts,
full seven-by-seven matrix, next-class counts, number of recordings with at
least one change and distribution of change counts per recording. Report the
number of recordings containing each target class on a change. No selection,
sampling, fitting, hyperparameter tuning, model evaluation or per-transition
identities are saved. Every raw episode's training membership is fixed before
the scan; training record later timesteps are not independent examples.

This is only train-source feasibility for a future prospectively frozen task.
No source-independent performance, physical timing, causal modality or product
quality claim follows. Existing fixed train and validation tasks remain intact;
a future test needs recording/operator/site separation and a predeclared policy
for all temporal windows and class prevalence.
