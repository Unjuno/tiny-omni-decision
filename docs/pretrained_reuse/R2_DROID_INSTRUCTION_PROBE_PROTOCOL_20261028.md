# DROID train-only instruction-conditioned probe

Freeze before execution: use the existing 275 sparse train recordings, their
pre-extracted eight-state sensor inputs and next-command targets. Read the
three language fields only at `target_index - 1`; join nonempty fields in
field order. Missing text is an empty input, never inferred from later steps.
Use the episode-wide normalized instruction audit solely to union all train
recordings sharing a string, transitively. Assign components to five folds with
seed 17 and no split components; save assignments and verify isolation. This
is a development probe, not a blind test or product evaluation.

Compare sensor-only with sensor plus fixed 256-dimensional SHA-256 hashed word
counts and an explicit missing-text indicator. Normalize the sensor dimensions
with each fold's training statistics. Fix both linear seven-class heads to 200
full-batch Adam updates, learning rate 0.01, weight decay 0.001, seed 17 and
natural label prevalence. No hyperparameter selection or checkpoint selection.
Save out-of-fold probabilities and report Accuracy, NLL, Brier and ECE for all
rows and the text-present subset, with class counts and per-fold metrics.
Inputs and output are hash pinned; no validation rows are selected or scored.
The grouping audit file contains metadata for the full source cohort but only
train IDs are joined. Sealed audit sets and media are not decoded, and prior
artifacts are not overwritten. All computation is CPU-only.

The two arms differ in input width and parameter count. Hashed words are a
low-cost information probe, not the required pretrained text representation,
and improvements cannot prove use of task intent. Shared texts are grouped but
recordings need not be independent operators or sites. A later, separate
frozen pretrained text encoder and independent data are needed to substantiate
model quality and multimodal complementarity.
