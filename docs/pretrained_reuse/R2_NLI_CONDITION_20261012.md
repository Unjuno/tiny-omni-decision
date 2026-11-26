# Approved fixed NLI comparison

Same256 questions/64 states and targets as the original condition probe; inputs
hash verified identical. Premise is unchanged state JSON. Hypothesis is exactly
`Candidate A has the lowest price.` with candidate, direction and field
substitution. Tests verify the transformation does not use target or add numbers.
Pinned NLI checkpoint label1 is verified as entailment; option probabilities are
softmax over its three entailment logits, temperature1. No training/calibration.

Execution PASS; useful conditional ranking NOT DEMONSTRATED:
Accuracy32.03%, NLL1.10984, Brier0.67407, ECE0.06791. Uniform3-way reference is
33.33% expected Accuracy, NLL1.09861, Brier0.66667. Opposite-condition answers
changed70/128pairs; both answers correct12/128pairs(9.375%). Response changes
alone are not correctness; independent uniform answers would have1/9 expected
both-correct probability. These descriptive comparisons are not significance tests.

The larger NLI reference did not solve this fixed numerical diagnostic. Lower
ECE/NLL than the confident all-B reference does not establish better reasoning.
Inputs fit(max68tokens). No prompt search or further tuning was performed.
No general architecture ceiling is inferred. This is a constructed diagnostic,
not blind or multimodal product validation; do not promote the model.

Source commit f9873fd. Local RTX3080 Laptop FP32, PyTorch2.6.0+cu124,
2.172sec timed load/inference; peak allocation414,903,296bytes. All weights
frozen; no cloud cost, sealed audit access or legacy artifact change.
Full outputs, revision and hashes are retained in adjacent JSON and external
artifact path. The next strategy must address learned conditional reasoning
and full-system parameter feasibility rather than further wording changes here.
