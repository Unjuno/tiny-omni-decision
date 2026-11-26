# Fixed soft-teacher replay comparison

Change only original-task hard labels to the existing training teacher distribution.
Retain hard context labels, ordered3072 examples, seed17, coverage parent, fresh
AdamW2e-5/.01, batch8,384updates, all encoder blocks/classifier, fixed final selection,
and identical context/original development. No temperature, seed or mixing search.
Choice maps IDs, score verifies ordered criteria, binary maps true/false. Reject
nonfinite/out-of-range values or sum error >.001; normalize admitted rounding.
Existing hard-replay artifacts remain read-only; new context-soft-replay outputs.

Seven mapping tests pass. Separate-process two-update smoke passed frozen-weight
and exact reload checks. This protocol precedes the main run; no quality result yet.
Compare all final and scheduled metrics with the fixed hard-replay run and parent;
do not treat synthetic teacher uncertainty as proven calibrated truth. The run
is a development intervention, not a product candidate or sealed-audit result.
