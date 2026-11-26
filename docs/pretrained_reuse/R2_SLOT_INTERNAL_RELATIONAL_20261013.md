# Fixed internal comparison: one slot interaction block

Same79 fit /20 internal-development videos, frozen features and text inputs,
seed17,50 epochs, AdamW .001/.01,8 pairs/batch,500 updates and7,900 question
presentations. Fresh initialization in each experiment. No epoch selection.
The additional block projects129 to64, applies LayerNorm and four-head self
attention (dropout0), projects back to129 and adds a residual before the original
candidate readout. It models token interactions, not supervised object tracks.

| Metric | Original head | Interaction head |
|---|---:|---:|
| Trainable parameters |115,008|148,481|
| CPU fit seconds |4.690|19.102|
| Train Accuracy |.778481|.816456|
| Train NLL |.468665|.393029|
| Internal-dev Accuracy |.425|.300|
| Internal-dev NLL |2.914194|2.784886|
| Internal-dev Brier |.977441|1.141853|
| Internal-dev ECE |.465143|.531306|
| Changed-target both correct |1/12|0/12|

Reject adoption: better training fit and slightly lower development NLL coexist
with worse Accuracy, Brier, ECE and pair correctness. The small40-question,
single-seed, observed internal-development set cannot establish a universal
architecture ranking. Added capacity alone did not resolve the problem in this
fixed experiment. Do not launch a hyperparameter sweep or inspect old validation
to rescue this candidate. Both artifacts remain preserved.

Command: `python -m scripts.run_slot_internal_relational`.
Source parent2adf95d plus the helper factory/runner changes committed in b673c90.
All computation was CPU; no encoder changed. Both heads' common initial layers
are initialized first with seed17, but the new block necessarily has extra weights.
This compares architecture packages, not parameter-count-matched hypotheses.
The helpers retain their original default model; existing runs were not rerun.

Outputs under C:/CodexArtifacts/pretrained-reuse-r2/:
`perception-occlusion-slot-internal-relational-01` and
`perception-occlusion-slot-internal-relational-eval-01`. They contain weights,
all51 training-history entries, configuration/input hashes, normal/mismatch/zero
predictions and exact checkpoint reload verification. The adjacent JSON preserves
evaluation metrics and hashes. The original257-video validation was not reopened.

The next step requires stronger evidence about input grounding/data coverage;
shared-Lens, merging and product promotion remain unjustified.

## Post-hoc paired uncertainty check

Saved example records match exactly between arms. Resampling20 whole video pairs
10,000 times with seed17 gives relational-minus-baseline Accuracy difference
-.125,95% percentile interval[-.300,.025]; NLL difference-.129308,
interval[-1.330742,1.089811]. Both include zero. See
`R2_SLOT_INTERNAL_PAIRED_COMPARISON_20261013.json` for hashes and method.
This does not establish a statistically resolved architecture ranking; rejection
means insufficient adoption evidence under the quality goals, not proof that the
architecture is universally worse. This post-hoc diagnostic does not provide a
new blind evaluation or license further tuning on these20 videos.
