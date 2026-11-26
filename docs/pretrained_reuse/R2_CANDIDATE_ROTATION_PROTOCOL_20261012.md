# Candidate-position rotation diagnostic

The previous completed134.5M decoder has weak seen-training fit and only12.60%
candidate-identity consistency under reversed options. Before more training,
measure whether balancing candidate positions materially removes the bottleneck.
No parameter change, training, temperature fitting or checkpoint selection.

Use the validation-selected step2048 of small-decoder-learning-01, SHA
0caea48debd72e7ca4fa7c4f31da99005d8e6d587f07f32a3f1e9603cc3203e7.
Reuse all1,024 questions in base and option_reverse arms of seed179 transfer.
These results/templates have already been seen: explicitly a development
diagnostic, never a new blind evaluation. No dropped difficult questions.

For each question with N distinct options, run all N cyclic left rotations,
keeping state/question unchanged and labels A..N assigned in rendered order.
Normalize candidate logits at temperature1, align probabilities to original
candidate identities, then arithmetic-mean all N distributions. No target is
used to render, rotate or aggregate. This removes a constant positional prior
but does not guarantee arbitrary-permutation invariance or correct reasoning.

Record raw probabilities, hashes, Accuracy/NLL/Brier/ECE by task and arm,
candidate reversal identity consistency, exact question counts and forward
counts, wall time and peak VRAM. Paired128-state bootstrap of Accuracy gain
over the saved single-forward result:10,000 draws,seed17,percentile95% interval.
Verify rotation0 against saved predictions before interpreting the comparison.
No tuned alternative aggregation. Keep failures. A useful gain requires higher
Accuracy and lower NLL; even then it is not a95% or product-quality claim.
No new learning budget follows automatically from this diagnostic.
