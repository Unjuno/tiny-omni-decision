# DROID dense train-window construction

Freeze this construction before reading source shards. Use exactly the prior
275 train recording IDs, pinned source receipt and parser. Verify all shard
sizes and SHA-256s and the prior fixed target per recording. Nonselected opaque
records may be read to identify them, but their numeric payloads must not be
decoded. No validation/test/audit records are selected; no image or video is
decoded. Existing datasets, checkpoints and outputs are read-only.

For every selected recording and every target step `t >= 8`, create exactly one
row: eight consecutive 14-value sensor observations at indices `[t-8, t)`, the
seven-class dominant-axis previous command at `t-1`, and the seven-class
target command at `t`, using the original `1e-4` threshold. Do not include
sensor observations or commands from `t` or later in the model input. Preserve
natural step prevalence and all seven options; do not filter change events,
duplicate rows or rebalance the held-out data. Record change/stable flags as
derived metadata only. Preserve original recording-group identity and the
previously fixed five outer train-group folds; no group may cross folds.

Write new train-only arrays and a private group/offset index in a new external
directory. Check 80,998 total rows and 7,442 change rows against the fixed
prior census, and verify all output hashes and per-recording counts. Refuse to
start unless available disk covers at least 1 GiB reserve plus 128 MiB. No
model fit or quality evaluation is performed in this construction step.

Rows from the same recording are correlated; neither the row count nor the
reused cross-validation folds imply independent operators or sites. Dominant-axis
label changes can reflect small command fluctuations; no physical timing or
causal multimodal claim is made. A later training procedure must be frozen in
advance, use recording-group folds, report natural-prevalence and change/stable
metrics and compare identical available inputs. Preserve all held-out sets.
