# Slot readout structural audit after fixed validation

Code inspected: `scripts/slot_feature_sampling.py` and
`src/tiny_omni_decision/trajectory_readout.py`, source bc43025.

The224 tokens are32 times x7 slots. Each contains128 slot coordinates and one
normalized timestamp. Slot index is not an input coordinate. The head applies
linear key/value projections independently to each token, candidate-conditioned
softmax attention, and a weighted sum. There is no explicit token-to-token layer.

For any permutation P of tokens, K(PX)=P K(X), V(PX)=P V(X). Attention weights
permute by P, and their weighted sum is unchanged. Consequently, independently
renumbering slots within every frame cannot change this head's mathematical output.
Floating-point summation can differ slightly. Sorting slots into persistent IDs,
then flattening them back into the same token set, is not a usable intervention.

This does NOT prove the head is time-blind: timestamp values and recurrent slot
content can influence scores. Nor does it prove information is absent from the
frozen slots. It establishes that correspondence encoded only by array position
is discarded. A future correspondence experiment must retain that relation in the
actual input or computation, and separately account for representation changes.

## Existing evidence and next decision

- Eight-train-video sparse greedy mask association already worsened IoU from
  .138612 to .086843. Do not repeat this matching heuristic search.
- Fixed50-epoch train readout fits some data, but validation Accuracy is .361868,
  NLL2.733249 and changed-target both-correct3/127. This is a failed quality gate.
- Neither result isolates whether localization, identity, condition grounding,
  small training coverage or head generalization dominates.

Before another trained head, use a predeclared training-only diagnostic to measure
whether the fixed endpoint uses timestamp values at all. It should reuse cached
features and weights, preserve every example, change timestamp values only, and
report probability changes as well as condition-pair metrics. No validation
selection, new checkpoint, backbone training or shared-Lens work is justified yet.
This diagnostic remains proposed, not executed; it cannot establish temporal
reasoning merely because predictions change under an input perturbation.
