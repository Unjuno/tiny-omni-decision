# Conditional token readout comparison

Prior pooled interaction results do not show a reliable video contribution.
Preserve the original 8-frame, 288-pixel FP32 frozen VideoPrism output; do not
average its 2048 space/time tokens before the attention condition.

Compare three heads: query-conditioned single attention readout, mean pooling,
and zero-video mean pooling. Candidate features remain the existing 1536-wide
frozen MiniLM representation. Query/key/value projections are 128-wide; fixed
normalized time/row/column coordinates have a learned linear key projection.
Each candidate's attended value is dotted with its query and divided by sqrt(128).
The mean control has the same query/value computation, without attention or
positions. Its unused key/position parameters must be reported as inactive,
not as equivalent effective capacity. Shared initial state and sample order.

Use the existing validated 3794 training choices/256 scenes and 3014 validation
choices/200 scenes, existing split/content gates, seed17, 24epochs, AdamW1e-3,
weight decay1e-3, effective batch64 and identical equal-type/scene weights.
Microbatch8 with gradient accumulation is allowed for all three conditions;
weight each microbatch by the full effective-batch denominator. No backbone
or text encoder gradients. No new data or sealed audit access.

Run separate small execution smoke before the full run. Preserve unpooled
features once per media in RAM, transport via the tested worker, verify every
global mean against the prior cache and record every token hash. Fail on
nonfinite values, source/hash/split mismatches or OOM. New output directories.

Earliest minimum validation macro question-type NLL selects best; preserve final
separately, require exact best reload. Save all histories, per-example predictions,
Accuracy/NLL/Brier/ECE by question type, sample/asset counts, train/validation gap,
time and VRAM. Evaluate scene-deranged video at selected checkpoint. Compare
paired scene-bootstrap differences with seed17/10000 replicates as in prior code.
This is observed development validation, not blind evidence or general video95%.
Measure additional trainable parameters and the full resident parameter total.
Do not adopt on aggregate Accuracy alone or automatically increase the budget.
