# Four-choice train-candidate option-only diagnostic

Freeze before constructing options. Use all 257 successfully saved train
candidates and their original positive targets, with the 11 previously frozen
classes. Do not use image bytes, validation labels, test labels or weights.
Require unique image IDs, target explicitly positive, three distinct explicitly
verified in-scope negatives and no positive/negative contradiction.

Process images in SHA-256 order under the fixed string `option-v1|17|<image>`.
For each image select three available negatives in ascending previously chosen
negative-count order; break ties by SHA-256 of `negative-v1|17|<image>|<label>`.
Order the resulting four options by SHA-256 of
`position-v1|17|<image>|<label>`. This rule is deterministic and uses the
entire candidate set to define the task; the later cross-validation is thus a
task diagnostic, not fully independent benchmark validation.

Use five folds from SHA-256 `fold-v1|17|<image>` modulo five. Each selected
candidate has a different normalized author identity; report fold sizes and
class counts. An option-only baseline learns per-class positive/negative odds
from the other four folds with add-one smoothing, then normalizes the four
odds. Record pooled Accuracy, NLL, multiclass Brier and 15-bin ECE, along with
uniform four-choice metrics and class role counts. Ties use supplied option
order. Do not tune this rule after observing results, discard difficult rows,
or use the resulting diagnostic for checkpoint or product selection.

Save task IDs, options and out-of-fold predictions privately in a new output
directory, with pinned input hashes. This task is training-candidate-only;
media ownership and content/split independence remain unproven, and this does
not create a new final evaluation or admit data for model training.
