# R2 time-aware gated fusion implementation smoke — 2026-11-23

## Outcome

Added a small causal fusion module for aligned frozen-feature snapshots and a
candidate-conditioned typed-option Decision scorer at
`src/tiny_omni_decision/time_aware_gated_fusion.py`. Its input is
`[batch, time, modality, hidden]`, with an explicit availability mask and one
common-clock timestamp per time step. At each step it learns a modality gate,
then updates a recurrent state through a feature-wise update gate. It returns
the final state, causal intermediate states, per-modality weights and update
gates for diagnostics.

The module contains 17,216 trainable parameters with hidden size 64 and four
modality slots. This count excludes upstream encoders and any per-modality
projection into the shared 64-dimensional feature width. The API requires the
caller to align asynchronous streams into causal as-of snapshots and to keep
the time unit explicit; it does not infer clock alignment or silently fill
missing inputs. A sample with no available evidence is rejected for the caller
to abstain.

The existing `FusionDecisionScorer` remains the fixed-vector masked control.
The new module preserves temporal order and should be compared against that
control only after an R2-admitted real task, fixed data split and candidate
contract are available. `TimeAwareGatedDecisionScorer` concatenates the final
temporal state with each supplied typed-option feature and returns masked
option logits. At hidden size 64, four modalities, 12 option features and a
128-wide option scorer, the complete feature-level scorer has 27,201
parameters; encoders and projections remain external.

## CPU smoke checks

`tests/test_time_aware_gated_fusion.py` adds eight CPU tests and verifies:

- finite output and normalized per-step modality gates;
- zero weights for missing modality slots and padding;
- future changes do not alter earlier recurrent states;
- masked NaN payloads cannot affect output;
- gradients reach available feature values;
- nonmonotonic valid timestamps and fully missing samples fail closed;
- the 64-wide, four-modality fusion module stays below 100,000 parameters.

Verification on the local CPU environment:

```text
python -m pytest -q tests/test_time_aware_gated_fusion.py tests/test_fusion_decision.py tests/test_temporal_native_decision.py tests/test_native_fusion_smoke_matrix.py
24 passed

python -m ruff check src/tiny_omni_decision/time_aware_gated_fusion.py tests/test_time_aware_gated_fusion.py
All checks passed
```

No model checkpoint, corpus, media, or validation labels were loaded. No GPU
or cloud compute was used. This is a mechanics smoke only; it provides no
accuracy, calibration, modality-necessity, task-generalization, or product
quality evidence. R2-M remains open until a rights-cleared task with valid
groups and a target requiring multiple inputs is admitted and measured against
matched unimodal and intervention controls.
