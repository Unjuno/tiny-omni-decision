# HARTH window availability audit

The existing prepared HARTH experiment targets sample index64 of each128-sample
window (`scripts/prepare_harth_subjects.py`). Both temporal and static full-window
readouts consume samples through index127. Therefore their historical quality
metrics describe delayed center-label classification, not immediate decisions
at the label timestamp.

Actual saved validation timestamp arrays were read only after verifying each
subject archive against its preparation manifest SHA256. All12,763 windows across
S013 (5,691), S027 (2,456), S019 (4,616) have end-minus-target time1260ms, within
floating-point rounding. Evidence:
`C:/CodexArtifacts/pretrained-reuse-r2/harth-window-availability-audit-01/result.json`.
No sealed audit payload, model weights or labels were modified.

A runtime adapter must expose full-window evidence only at or after its end
(and account for actual processing delay separately). Its output must identify
the target time independently of availability time. Setting availability to the
window center would leak future samples into a purported causal decision.
This1260ms is observation lookahead relative to the target, not measured compute
latency. Whole-episode VIDIMU features likewise become available after the whole
episode; existing synchronization uncertainty still applies.

No retraining, relabeling or new quality claim follows from this audit. Historical
scores remain valid in their delayed/offline task scope. A current-time causal
quality evaluation would require a separately fixed task and labels, and cannot
be inferred from the existing scores.
