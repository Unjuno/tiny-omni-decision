# DROID previous-command plus sensor grouped-CV diagnostic

Freeze before fitting. Reuse exactly the 275 already observed train recordings,
the original five outer grouped folds and previously extracted sensor sequence,
previous commanded action and labels, with all source hashes verified. Use no
validation features, validation labels, audit data, future commands, video,
encoder inference or GPU. Only the first 275 train-first rows may be copied
from the original memory-mapped feature container.

Compare two seven-class L2 softmax heads using the same previous-command input:
one-hot previous command (7 inputs), and the same one-hot concatenated with
the existing final-state plus seven coordinate differences (112 sensor inputs).
They have 56 and 840 trainable weights respectively, including biases. This is
a sensor-ablation comparison with different parameter counts, not evidence of
an equal-capacity architectural advantage. Both use the existing pinned readout,
outer folds and four inner grouped folds; select L2 only within each outer
training partition from `[100, 1000, 10000]` by macro NLL, seed 17, max 2000
iterations, and fit standardization only on the relevant training rows.

Record both OOF predictions, inner scores, outer train/test IDs, model hashes,
convergence, Accuracy, macro Accuracy, NLL, macro NLL, Brier and ECE-10. Report
previously computed add-one transition-table OOF metrics as an additional
history-only baseline. Predeclare primary comparison as history-plus-sensor
minus history-only macro NLL; separately report target-defined changed-command
cases and counts as descriptive diagnostics, not a selection criterion.
No grid expansion, checkpoint selection on held outcomes, training of a final
product model or sealed evaluation. Existing predictions/checkpoints are
read-only; new models and predictions go to a new external directory.

This is development evidence on 275 previously inspected train recordings, not
an independent operator/site or blind evaluation, and does not prove temporal
causality, multimodal benefit or product-quality performance. Unknown physical
time gaps and raw angular differences are preserved without physical claims.
