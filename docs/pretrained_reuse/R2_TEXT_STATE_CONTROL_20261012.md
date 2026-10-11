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
