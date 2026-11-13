# Fixed lexical-transfer comparison

Before inference: compare numeric-only selected1408
SHAf287bb0138fd0f264331cd2fc983567bc68323a85a25cda467387fffae7bc93b
against lexical-range selected1408
SHAfccee21e0e5ffbaffdee899a65fe6538f52f3b788752a56b6b5b6388cad15abb.
No final weights, fitting, new seed training or re-selection. Same22.7M base,
processor,FP32 and original/curriculum training budget and order; previously
verified question-only training difference.

Generate256new states with seed127,8questions/state. Five2,048-question arms:
original control;7*x+19 interpolation;13*x+29 mixed extrapolation; novel wording
with original values;13*x+29 plus novel wording. Numeric transforms match the
previous diagnostic but states are new; this is follow-up development, not blind.

New templates, fixed before inspecting outputs:

- `Find the minimum/maximum <field>; give the item's identifier.`
- `How many records satisfy risk <= <threshold>?`
- `Considering only records with risk <= <threshold>, return the identifier
  with minimum cost. If no record meets the condition, return none.`

Keep row/option order, labels, state IDs and all questions consistent across arms.
Reject group/content overlap against all prior generated train/validation/probe
corpora; reject normalized paraphrase overlap with training or previous probe
wording (retaining the original control templates intentionally). Verify answers
independently and save hashes before inference. No seed retry after a failure.

Use existing five-arm evaluator,10,240inputs/model,512token no-truncation guard,
all probabilities and per-task Accuracy/NLL/Brier/ECE. Compare lexical minus
numeric with256-state paired bootstrap10,000replicates,seed20261012,percentile95%
intervals. Report all arms, including regressions. Single seed, reused task
families and development selection; no multiplicity/selection correction.
If wording improves but numeric/control deteriorates, report the trade-off.
No old transfer input rerun and no product promotion from this test.
