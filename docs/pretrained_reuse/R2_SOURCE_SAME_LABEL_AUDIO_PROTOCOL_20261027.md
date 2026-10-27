# Train-only CV: same-label audio mismatch diagnostic

Use the same three fixed out-of-fold joint checkpoints and 2,593 previously
frozen training-source examples. Match each recipient's coalesced sound IDs to
the pinned `all_train.json` source event annotations. A record is eligible
only when every coalesced annotation has exactly one consistent source label ID
and its fold has another video with an eligible record of that label. Report
all unmatched/mixed/missing counts; never silently drop them or claim the
eligible subset represents all cases. Label IDs are used solely to choose
diagnostic donors and never supplied to the model.

Within each held-out fold and label, sort donor video IDs; map each video to the
next video cyclically and use that donor video's lexically smallest eligible
record ID. Verify the donor has a different video ID and the same source label.
Compare the saved original audio with this donor on exactly the eligible
records, preserving all original checkpoints, visual inputs and targets.
Report group-paired accuracy and set-NLL differences, 10,000 bootstrap draws,
seed 17, visible/none splits and coverage by label/fold. Check original scores
against saved predictions before accepting the outcome; use a fresh output and
save the donor mapping hash. No additional fit, model selection, official
validation or sealed audit.

Source labels can be broad (for example generic object sound), and different
video groups may share participants, sessions or media similarity. Same-label
donors do not guarantee matching physical source or speaker, nor prove causal
audio-visual binding. These are nonblind, correlated cross-validation
interventions, not independent generalization or product quality results.
