# ADL: cross-activity timestamp metadata

The two fully verified activities were checked in both train-to-validation
directions on sensor 1. Light-off train (290 events) against take-from-stove
validation (75 events), and take-from-stove train (293 events) against light-off
validation (79 events), each had **zero exact timestamp matches**. Minimum
nearest absolute differences were 85,205.646 and 83,818.131 respectively, in
the original stored units. Both activities had zero unassigned rows.

This is only an exact stored-timestamp check, with no waveform or duration
comparison. Units, recording boundaries and independent sessions have not been
verified; this result cannot establish absence of data leakage or justify
training. It covers two activities and one sensor per activity.

The [fixed procedure](R2_ADL_CROSS_ACTIVITY_TIME_PROTOCOL_20261027.md) was
recorded before execution. The receipt at
`C:/CodexArtifacts/pretrained-reuse-r2/adl-cross-activity-time-20261027-attempt-02/result.json`
has SHA-256 `18e626144b0821b6ff9415eedf3df07cdd3e2b820fd0f36eb98eeba46c064ee9`.
The first execution failed before producing a receipt because Python ZipInfo
objects cannot be sorted directly; filename sorting was corrected and covered
by a regression test before the successful second attempt. Five targeted tests
and Ruff passed. No model, training, validation scoring, or audit ran.
