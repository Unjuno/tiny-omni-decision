# R2 native semantic-role supervision opportunity — fixed 2026-10-14

## Question and bounded scope

After the native mask corpus and derived feature bank pass their complete
admission/replay/independent audits, inspect whether the retained candidates can
be paired with source training-box proxies for semantic supervision. No readout
fit, QA predictions, dev/sealed metrics, new model/GPU inference or runtime GT.
This is a **training-only geometric opportunity census**, not role recognition.

All8 previously known train clips, all32 fixed times, every raw1,024 mask and all
18,441 NMS survivors. Read the original pinned `train_annotations.zip` only after
feature admission; no validation archive. Parse exact upstream grounded hidden,
cover and group track IDs with the existing tested suffix parser. Cover is a
member of the group, not an additional independent object. Unique source-track
count uses(videoID,trackID); proposals/cases/frames are not independent examples.

For each group member and hidden source object at every fixed time, use the
existing range-limited source bbox interpolation. Preserve exact/interpolated
brackets, distance to nearest annotation and `is_masked` brackets. No extrapolation
before/after annotation ranges. Missing source boxes stay missing, not detections
that failed. Source annotation start is not physical birth; masked flags are not
verified visibility. Keep every unavailable case in the saved census.

Measure raw/NMS maximum **bbox IoU**, unchanged threshold>=0.5, against each source
box. Record all target cases, both maxima/indices and matching NMS counts; no
best-mask selection enters inputs, extraction, runtime seeds or fitting. For each
clip/time separately count whether hidden and designated cover both have available
source boxes and at least one retained proposal clearing the same geometric gate.
This is co-occurring geometric support, not a measured relationship/memory decision.
Report hidden/group/cover counts with available/missing denominators per clip and
aggregate. No visibility denominator, detector ceiling, model-quality claim or
reinterpretation of unavailable/hard cases. All8 scenes are already known training
diagnostics; do not treat them as fresh validation or independent held-out evidence.

Hash-bind native result/config/banks, complete native audit, feature primary/fresh
results/config/archives and independent feature audit **before opening labels**;
verify every input/annotation/helper hash again at exit. Read-only sources, fresh
output file, normal Python assertions enabled, NumPy RNG unchanged. CPU only;
record actual body time/script hash and retain incomplete errors. No cloud cost.

This census determines which supervision is actually supported before defining
any new semantic readout experiment. GT source IDs/boxes are permitted as posthoc
training supervision, never runtime observations. No tuning of threshold/selection
after the results, no retry of closed distance/cosine routing or failed QA heads.
The95–98% four-modality and joint/OOD product objective remains unachieved.
