# Training-only token readout fit diagnostic

The completed 24-epoch token comparison has weak train fit as well as weak
validation quality. Diagnose ability to fit a small fixed training subset before
another validation search. This is a memorization/optimization diagnostic,
never a benchmark success or a candidate-selection result.

Take the first64 unique training choices in the existing verified corpus order,
without looking at labels or performance to select them. Record exact IDs,
task/label/scene counts and hashes. Re-extract only their unique video assets
using the same pinned frozen8-frame encoder and verify pooled-cache parity.
The existing shared input validator reads development split metadata for integrity;
no validation examples are forwarded or scored, no sealed audit is opened.

Compare fresh seed17 attention and no-video heads with identical initial weights.
Keep dimensions, coordinates, FP32, AdamW1e-3/weight-decay1e-3 and existing
equal-question-type/scene weighting. Use all64 examples per optimizer update,
microbatch8, 512 updates, no early stopping and no checkpoint selection.
Evaluate these same64 training examples at step0 and every32 updates.
Save final weights separately by condition and require exact reload.

Log weighted CE, train Accuracy/NLL/Brier/ECE by task, gradient norms, wall time,
VRAM, total encounters and unique samples/assets. Save final per-example
probabilities and hashes. For attention, record mean/min/max normalized token
entropy on these training examples, and gradients by named parameter.
Finite-value failures stop the run. Do not extend steps automatically.

High fit only shows that this subset can be memorized. Failure to fit in this
budget does not prove a base-model capacity limit. A no-video fit would show
that training-subset success alone is insufficient evidence of video reasoning.
