# Source candidate: select-every task versus one-choice scoring

All 2,593 fixed train-source records instruct: "Select every visual candidate
that caused the audio event; if none is visible, select none_visible." The
existing softmax readout and reported hit@1 select one candidate and count it
correct when it belongs to the target set. That is a useful ranking probe,
but it does not evaluate the full select-every instruction. In particular,
the grouped NLL rewards the total probability on any correct member; it does
not require identifying every member. Earlier reported hit rates remain valid
only as hit@1, not exact-set task accuracy.

The following diagnostic derives exact-set accuracy from the existing
single-argmax output; it does not train or select a new model. It checks the
original annotation instruction, targets, candidate identities, prediction
hashes and recorded hit against reconstructed argmax.

| Epoch | Hit@1 | Exact set with one-choice output | Single-choice exact-set ceiling on these records |
|---:|---:|---:|---:|
| 4 | 1,365 / 2,593 (52.64%) | 665 / 2,593 (25.65%) | 1,491 / 2,593 (57.50%) |
| 8 | 1,479 / 2,593 (57.04%) | 753 / 2,593 (29.04%) | 1,491 / 2,593 (57.50%) |
| 32 | 1,454 / 2,593 (56.07%) | 803 / 2,593 (30.97%) | 1,491 / 2,593 (57.50%) |

The ceiling is a mathematical property of always emitting exactly one option
on these labels, not a bound on the backbone, probability vector, task or
other decision rules. A set-valued readout and explicitly defined independent
or joint option probabilities are needed to satisfy the stated task; neither
the appropriate training loss nor calibration has yet been validated. A fixed
threshold on existing softmax scores would be a different decision rule and is
not evaluated here. Future comparison must predefine set metrics and calibration
semantics, retain `none_visible` exclusivity, use clean grouped validation and
compare matched unimodal models. The data remain nonblind oracle-track internal
development with unresolved contributor grouping; these results do not measure
product performance. The 95% overall goal is not reached or redefined here.

The analysis used previously saved predictions and train annotations only;
no sealed audit, new training, GPU use or data download occurred. Output:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-set-task-audit-20261027-attempt-02/result.json`,
SHA-256 `925cbfd81c3fdfd1f5cd8a11b7897a8670021122d01401283dc6ff6b11343d79`.
