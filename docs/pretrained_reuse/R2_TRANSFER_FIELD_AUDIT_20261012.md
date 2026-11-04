# Correction: transfer family4 does not rename state fields

Read-only check of the actual10240-row immutable transfer file (SHA in JSON)
compared8192 family1–4 rows against their corresponding family0 row in the same
numeric regime. Every paired state string, option list, target, oracle, task and
latent group is identical. No data or model inference was changed.

Family4 changes1536 of its2048 question strings: extremum and temporal-order
questions gain domain adjectives plus the explicit preamble:
`Use cost as price, risk as danger, and time as timestamp.`
The remaining512 count/conditional-selection questions are identical controls.
State keys remain `cost`, `risk`, `time` (and ID), not `price`, `danger`, `timestamp`.
Numeric expansion between regimes is separate and remains7*x+19.

Earlier protocol/results wording about field naming could incorrectly imply a
state-schema change. It should read **question wording changes: domain adjective
and explicit alias preamble**. Existing scalar metrics and probabilities remain
valid and unchanged; their causal interpretation was too imprecise.

This narrows the unresolved cause to question-side changes, without isolating
the adjective from the alias preamble or its extra tokens. It does not prove
that one word is the sole bottleneck, and cannot justify rewriting or filtering
these evaluation examples. No training was launched from this correction.

Reproduce the constructor invariant with
`python -m pytest tests/test_vocabulary_transfer.py -q`.
The actual-file audit keys each row by(variant,base_id), pairs against base-family0
within the same numeric regime, and checks the six fields listed above exactly.
The result JSON records all counts and the immutable input digest.
