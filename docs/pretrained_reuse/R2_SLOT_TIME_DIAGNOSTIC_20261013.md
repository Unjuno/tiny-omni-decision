# Fixed endpoint timestamp sensitivity (training only)

Executed `python -m scripts.diagnose_slot_time` on CPU using all99 train videos,
198 questions and the unchanged epoch50 checkpoint. No validation was opened,
no training was performed, and no candidate checkpoint was selected.
Output: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-slot-time-diagnostic-01`.
Input hashes, checkpoint/script hashes and full metrics are in the adjacent JSON.
All probabilities are preserved in the local predictions.json.

| Timestamp input | Accuracy | NLL | Brier | ECE | Changed predictions |
|---|---:|---:|---:|---:|---:|
| Original | .757576 | .443811 | .286884 | .064477 | 0/198 |
| 1-t | .737374 | .469835 | .294009 | .052301 | 14/198 |
| Zero | .747475 | .458557 | .290544 | .053197 | 8/198 |

Changed-target both-correct counts are9/53,5/53 and6/53 respectively.
Maximum absolute probability changes are .303868 and .161550.

The fixed head is sensitive to its explicit timestamp coordinate. The hypothesis
that it ignores that coordinate completely is contradicted on these training
examples. This does not demonstrate temporal reasoning or validation transfer.
The recurrent slot vectors retain the original chronological video information:
changing their appended timestamps is NOT video reversal, and setting timestamps
to zero does NOT remove all temporal information. The perturbed input combinations
can be out of distribution. Neither control isolates object-identity failure.

Tests verify only the last coordinate changes, input tensors are not mutated,
normal mode is a copy, and unknown modes fail. Four tests and Ruff passed.
The pre-existing overfitting/poor-validation result remains the quality decision;
do not replace it with these training metrics or tune on the old validation.
