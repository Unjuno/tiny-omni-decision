# Balanced source readout: same-label audio substitution

Use the fixed seven-epoch, seed-17 class-balanced joint checkpoints and the
existing 2,581 matched train-source records from the earlier same-label audio
intervention. Reconstruct the exact donor map from its hash-checked records;
require its SHA-256 to match the prior receipt, and verify same fold, distinct
video and identical source label. Labels choose donors only and never enter the
model. Keep videos, candidates, labels, decisions and checkpoints fixed.

Replay original conditional and none probabilities and exact decisions against
the saved balanced-set predictions within absolute 1e-6 before accepting each
record. On the same matched records, substitute the other video's audio feature
and evaluate original versus substituted exact-set accuracy, candidate macro
NLL/Brier, none NLL/Brier and decisions on none/one/multi strata. Save records
and per-video-group paired percentile bootstrap intervals (10,000 draws,
seed 17), coverage and unmatched count; no threshold, checkpoint, seed or
model selection. Input identity is different video, not proven different
participant or physical source; labels are broad and can be noisy. Original
versus altered differences indicate sensitivity, not causal AV binding or
proof that both modalities are needed. Existing train-source data are nonblind,
with oracle tracks, correlated training folds and unresolved group identities.
Do not open sealed audit or rerun official validation. No training, GPU use,
external data acquisition or overwrite. Fail on mismatched hashes, masks,
record IDs, nonfinite scores, disk below 1 GiB or existing output.
