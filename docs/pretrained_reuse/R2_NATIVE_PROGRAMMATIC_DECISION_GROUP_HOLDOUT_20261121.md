# R2 programmatic decision state-group holdout

## Preregistered scope

This is a separate, TRAIN-only generalization probe on
`text:programmatic_decision_v1`. It follows the completed condition-relevance
group holdout but does not reuse that holdout for fitting or selection. The
source contains 8,192 generated decision rows across eight tasks and 1,024
underlying state groups. Each group contributes one row to each task.

The fixed 2,048-update multimodal sampler previously touched 511 programmatic
state groups, leaving 513 untouched. This experiment hash-ranks those 513
groups using seed 17 and split ID
`native-text-programmatic-group-holdout-v1`; the first 256 are held out and
the remaining 257 untouched groups stay available for fitting. The fit split
also contains the 511 groups seen by the previous sampler, so no inference is
made that they are new data. All rows from a state group stay together. The
expected split is 6,144 fit rows and 2,048 holdout rows, with exactly 768/256
rows per task. A preflight receipt must verify these counts and disjoint row,
group, and observation IDs before training.

The experiment keeps the previous fixed-feature readout and optimizer schedule
unchanged: fresh seed-17 readout initialization, frozen text state and option
features, 2,048 AdamW updates, four microbatches per update, 0.001 peak LR,
62-update warmup, cosine decay, terminal checkpoint only, and no holdout-based
selection. The only experiment-specific settings are the new source/split ID,
the predeclared group counts, and an output subdirectory on local C:.

## Evaluation contract

Evaluation uses only the 2,048 held TRAIN rows. It reports each of the eight
tasks and their equal-task macro, plus an input ablation that zeroes the text
state while keeping candidate-query features fixed. Candidate counts vary
between four and five, so score arrays are padded only for storage and metric
code must ignore padding via the existing `-1e30` sentinel.

No counterfactual donor-target arm is included. Programmatic questions encode
numeric thresholds, and `filter_min_cost` has sparse compatible question and
option-set donors; applying a different group's answer would not preserve the
original question's target semantics. The original and zero-state arms are
paired by state group. These results diagnose this generated task family only;
they do not establish general text competence or product quality.

The corpus and frozen feature banks are pinned to the already staged C: copy.
Only TRAIN rows are read. Official validation and sealed audit remain closed.
The condition-relevance holdout and prior sampler are not retuned. Any failed
attempt is recorded separately; no partial run is resumed as an exact training
resume. All outputs live below
`C:\\CodexArtifacts\\pretrained-reuse-r2-local\\native-text-state-group-holdout\\runs\\programmatic-v1`.

## Status

Protocol/config/code are under implementation. Results are added only after a
passing preflight and completed fixed-terminal run. No quality claim is made
from the corpus row count alone.
