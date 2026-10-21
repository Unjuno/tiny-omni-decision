# Typed Decision JSON v1

The experimental `DecisionOutput.to_decision_json` method serializes already
computed option probabilities into a stable, caller-friendly JSON object. It
does not run a model, change scores, calibrate probabilities, or add model
parameters. The legacy `to_json` format remains available and unchanged.

```json
{
  "schema_version": 1,
  "request_id": "route-choice-17",
  "decision_type": "choice",
  "selected_option_id": "slow",
  "confidence": 0.7,
  "options": [
    {"id": "continue", "text": "Continue", "probability": 0.1},
    {"id": "slow", "text": "Slow down", "probability": 0.7},
    {"id": "stop", "text": "Stop", "probability": 0.2}
  ]
}
```

`decision_type` is one of `choice`, `yes_no`, or `ordinal_score`. The caller
supplies stable option IDs and their text in the same order as the model's
probability vector. The serializer rejects duplicate/empty IDs, missing text,
invalid probability distributions, mismatched option counts, invalid selected
indices, and a selected ID that disagrees with the probability maximum. A
`yes_no` decision must contain exactly two options. The sum of probabilities
must be one within `1e-5`. `ordinal_score` also requires finite, strictly
increasing numeric `ordinal_values` in option order and returns the
probability-weighted `expected_score`.

This contract does not define abstention, calibration guarantees, or input-media
transport. Ordinal values are supplied by the caller and do not define task
semantics; these require task-level policy and evaluation. A well-formed JSON
response is not evidence of model quality.
