# Fixed adapted-text state control

Protocol recorded before execution: fixed selected epoch-3 Cross-Encoder,
all 2,540 existing development validation questions, candidate definitions kept.
Compare intact state versus an empty state using the same checkpoint, processing,
precision and batch32. No weights, temperatures, selection or training change.
Intact predictions must reproduce saved best probabilities within 1e-5 before
interpreting the control. Report Accuracy/NLL/Brier/ECE and prediction changes.
State removal changes the input distribution; worse performance is evidence
of state reliance in this test, not sufficient evidence of correct reasoning.
No new blind evaluation or product promotion. Local GPU inference only.

## Measured result

Both processes exited 0. Intact reload maximum probability error versus saved
best was 1.97e-6, below the predefined 1e-5 tolerance; Accuracy was identical.

| Input | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Intact | 58.11% | 0.93779 | 0.53431 | 0.04276 |
| State omitted | 45.39% | 1.11017 | 0.63743 | 0.08468 |

1097 of 2,540 predicted answers changed. The 12.72-point decrease and
worse NLL/Brier show useful state reliance under this intervention. The model
also answers many items without state; question/option priors and dataset
regularities remain plausible. This does not prove faithful computation,
arithmetic ability, unknown-task generalization or causal world understanding.
No additional training or model-selection decision follows from this control.

RTX 3080 Laptop, PyTorch 2.6.0+cu124, FP32, batch32 candidate pairs,
Transformers local checkpoint, 9,015 pairs each. Intact run wall22.46sec,
omitted4.37sec (includes loading/inference; shorter input affects latency),
peak PyTorch allocations384,988,672 and282,849,280bytes. No cloud costs.
Source commit e37c7d1, base revision and artifact hashes in adjacent JSON.
Legacy files and selected checkpoint unchanged; no sealed audit access.
