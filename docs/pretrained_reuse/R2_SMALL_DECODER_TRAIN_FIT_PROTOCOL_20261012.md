# Fixed consumed-training diagnostic

While the main run is active, freeze a post-run diagnostic of the first1,024
scheduled training questions (512original+512curriculum), unchanged order/options.
Use the already-selected checkpoint only after completePASS/reload and verify
its selected step has consumed at least these1,024questions. No selection or
further training depends on this diagnostic in the current run.

Report per-source/task Accuracy/NLL/Brier/ECE and predictions with the same
prompt/scoring pipeline. This asks whether the selected model fits already-seen
examples. Training curriculum includes expanded ranges/word variants whereas
validation curriculum is narrow: a difference is NOT a same-distribution
generalization-gap estimate. Rolling CE likewise is not that estimate.
Retain low training fit as evidence of fitting/optimization/representation limits
within this fixed run; do not uniquely attribute it to model capacity or establish
that more training cannot help. No original probe or sealed audit is read.
