# Question-side2x2 diagnostic

Question: does the domain-family failure depend on the alias preamble, the
remaining wording family, or their interaction? The wording factor includes
syntax as well as domain adjectives; it is not a single-token causal intervention.
No hypothesis is assumed true in advance.

Use the same immutable128-state transfer inputs, two numeric regimes,1024 base
questions per regime, and the same fixed control1664 and coverage1792 weights.
For extremum/time questions create four predeclared cells:
- generic: original family0 question, no prefix;
- generic_alias: exact alias prefix + original family0 question;
- domain: original family4 question with the exact prefix removed;
- domain_alias: original family4 question unchanged.
Prefix: `Use cost as price, risk as danger, and time as timestamp. `
Count/conditional questions stay byte-identical in all cells as retention controls.
State/options/targets/oracles/order remain unchanged. Do not simplify options.
All8192 rows per model retained; same inference code,batch8,FP32,512tokenfailgate.

Freeze inputs/hash before evaluation; verify all symbolic targets. No training,
calibration, prompt search, alternatecheckpoint/seed or new audit. This is reused
failure-guided development data. Report all metrics by cell/type, paired state
intervals for prefix effects within each wording family and wording effects
within each prefix level.10000draws,seed17,percentile95%; no multiplicity correction.
Also report opposing-condition both-correct counts. Do not present mean across
repeated cells as independent observations or general quality.
Existing generic/domain_alias cells must reproduce saved logits/probabilities
within1e-6 and unchanged argmax; record maxdelta and stop interpretation otherwise.
The new input file and all new outputs use separate directories. No production
adoption or further training from a favorable cell alone.
