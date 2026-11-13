# Post-vocabulary fixed-state transfer design

Freeze this design while comparison-vocabulary training is active, before any
new candidate inference. Do not alter its run, selector or training data.

Compare the old lexical-range selected1408 checkpoint
SHAfccee21e0e5ffbaffdee899a65fe6538f52f3b788752a56b6b5b6388cad15abb
with the new run's earliest-minimum validation-NLL selected checkpoint, only
after completedPASS2048, frozen hashes and exact reload succeed. Record the new
selected step/hash before inference. Do not use final weights unless selected by
that existing rule. No further tuning based on this diagnostic.

Seed167,128unused states, eight original task questions per state. Ten paired
conditions: original numeric range or7*x+19, crossed with five lexical families:
minimum/maximum, smallest/largest, lowest/highest, least/greatest, and domain terms.
The first four families are now trained vocabulary, not unseen-word evidence.
The last uses cheapest/priciest for cost, safest/riskiest for risk, earliest/latest
for scalar time; it is a separate semantic-transfer arm, not a one-word causal
comparison because field naming also changes.

New fixed templates:

- Ordinary families: `Return the ID associated with the <word> <field> among these items.`
- Domain family: `Use cost as price, risk as danger, and time as timestamp.
  Return the <domain adjective> item's ID.`
- Count, unchanged across lexical families: `Count the items whose risk is <= <threshold>.`
- Conditional, unchanged across lexical families: `Under a risk ceiling of
  <threshold> (inclusive), return the minimum-cost item ID, or none if no item is eligible.`

Positive affine transforms preserve labels/ordering and transform thresholds.
Keep item count, all options/positions and all questions. Verify oracle, reject
state/content overlap with previous generated datasets. Reject exact normalized
template reuse except intentional within-this-study paired arms. Publish corpus
hash before inference.128states ×8questions ×10conditions =10,240condition rows,
not10,240independent examples; count/conditional text repeats across lexical arms.
Record distinct content count and do not pool repeats into a headline test size.

Same local FP32 processor/weights; no truncation. Report all per-condition/task
Accuracy/NLL/Brier/ECE, direct opposing-condition pair correctness, all prediction
probabilities, selected checkpoint accounting and runtime/VRAM. Paired128-state
bootstrap10,000replicates,seed20261012,percentile95% Accuracy intervals for new minus
old. Keep each arm separate; no best-vocabulary selection or temperature fitting.

This is known-failure-guided synthetic development, not sealed/blind audit or
general95%multimodal evidence. Domain terms have no prior training guarantee beyond
this local curriculum (base pretraining is unknown). Count/conditional retention,
numeric range and calibration regressions must remain visible. Existing weak
benchmarks and checkpoints stay intact. No new training is authorized here.

## Clarification added after actual-file audit (2026-10-12)

The earlier field-naming description refers to the question's alias preamble,
not renamed state keys. All state fields/values are identical across vocabulary
families within each numeric regime. No recorded metrics or inputs changed.
See [field audit](R2_TRANSFER_FIELD_AUDIT_20261012.md).
