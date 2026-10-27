# DROID train-window change-label persistence

This post-hoc descriptive check uses the existing 80,998 dense windows from
275 training recordings, not validation or a new model run. The pinned source
summary and `group`, `step`, `previous`, and `target` arrays were rehashed.
All 80,723 consecutive within-recording pairs satisfy next `previous` equals
current `target`; there were zero mismatches. The 7,442 windows where the
discretized target differs from the previous command define the change events.

For each horizon, include only changes with a window that many steps later in
the same recording. Count whether **every** intervening target remains the new
label, and separately whether the final label equals the old or new label.

| Following steps | Eligible changes | All following labels remain new | Final label is old | Final label is new |
|---:|---:|---:|---:|---:|
| 1 | 7,427 | 6,680 (89.9%) | 159 | 6,680 |
| 2 | 7,402 | 6,102 (82.4%) | 232 | 6,142 |
| 4 | 7,332 | 5,022 (68.5%) | 404 | 5,106 |
| 8 | 7,217 | 3,133 (43.4%) | 838 | 3,356 |

The final-label counts can include intervening changes and are not the same as
continuous persistence. Most changes persist for one next step, while fewer
remain unchanged for all eight; this rules out treating every change as an
immediate one-step reversal in this sample. It does not establish physical
event duration, measurement noise, intent, correctness of the discretization,
or downstream prediction quality. Events overlap in time and recordings, so
they are not independent observations. No time-in-seconds claim is made.

The aggregate file is
`C:/CodexArtifacts/pretrained-reuse-r2/droid-change-persistence-20261028-attempt-01.json`,
SHA-256 `7ac08a5a1530c7feca8a4d18ab7fb6f51042cd83479b5848d4753ff8f34acb85`.
The source summary SHA-256 is
`31abc71b9d31f69e906c6e98ee712c82384dc90bbd5ab1a9a6122b0c12953982`.
The analysis was a local inline NumPy invocation; it used no GPU, media,
validation, sealed audit or optimizer, and does not select a checkpoint.
