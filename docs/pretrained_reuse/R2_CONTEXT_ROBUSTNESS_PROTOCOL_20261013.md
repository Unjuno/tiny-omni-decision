# Context robustness: fixed next comparison

Motivation: the completed prefix factorial found quality loss from answer-preserving
context. Do not train on that diagnostic or strip prefixes from runtime requests.
Use the new context-robustness-corpus-01 with seed211/223 state-disjoint train/dev.
Three views of each extremum question: plain, irrelevant context, necessary context.
Necessary context explicitly names the field; the remaining question does not.
Development context templates differ from train. Comparison words remain familiar;
this is not an experiment claiming general domain-vocabulary recovery.

Corpus:512train states x6questions x3views=9,216rows;128development states=2,304rows.
All targets verified against the numeric state. Group/content overlap checks cover
the exact previous files listed in the manifest. Actual tokenizer max147tokens.
Generator tests cover deterministic output, unchanged targets/options, necessary
field placement, independent states and held-out context templates (3passed).

## Predeclared learning comparison

- Both arms start from coverage best-delta b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5,
  optimizer reset, same pinned22.7M base; no strict-resume claim.
- Same3,072 underlying training question IDs in the same seed17 shuffled order.
- Plain control uses the plain view of every question. Context candidate uses one
  view per question, with exactly1,024plain/1,024irrelevant/1,024necessary rows.
  Assignment and ordering must be saved before training; no repeated question IDs.
- Last encoder block and classifier only; measure actual trainable parameter count.
  AdamW fixed2e-5, weight_decay .01, batch8,384updates, seed17, no scheduler search.
- Fixed final checkpoint comparison, no selection from development. Save initial,
  final, all scheduled curves and per-example probabilities. Development at0,128,
  256,384; report each context separately with Accuracy/NLL/Brier/ECE, opposite
  condition pairs and candidate-order invariance. No old transfer re-evaluation.
- Each arm in its own new directory; separate-process smoke before real fitting.
  Verify frozen tensors, finite gradients, input hashes, identical initial weights,
  exact reload and source preservation. Stop for execution faults, not poor quality.

The candidate changes the rendered training view, not capacity, base questions,
numeric values, targets, options or optimizer budget. Some fields differ in how
they are expressed in necessary context; that is the intended treatment. Report
trade-offs on plain questions and held-out templates. No adoption merely from
aggregate gain, no additional seeds/budgets or prompt sweeps in this comparison.
Observed synthetic development results cannot satisfy full Omni quality gates.

This commit prepares and validates the actual corpus. Training is not yet started.
