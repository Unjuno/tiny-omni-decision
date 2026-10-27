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

## Post-hoc LSTM error breakdown

The saved raw OOF probabilities from the fixed 50/200-epoch LSTM curve were
joined to the same train windows by row identity, after verifying the curve and
dataset hashes. Of 7,442 changes, 110 have no complete four-step follow-up.
The remaining 7,332 split into 5,022 whose next four labels all remain new and
2,310 with another label change during those four steps. These groups use
future labels only for retrospective analysis; they are not model inputs or a
training/evaluation selection rule.

| Group | Rows / recordings | Correct new label, epoch 50 | Correct new label, epoch 200 | Any change detected, 50 / 200 |
|---|---:|---:|---:|---:|
| Persists four steps | 5,022 / 274 | 128 (2.55%) | 238 (4.74%) | 310 / 598 |
| Changes again within four | 2,310 / 254 | 27 (1.17%) | 61 (2.64%) | 138 / 262 |

Both subsets contain correlated and overlapping windows; the recording sets
overlap too. Low changed-command accuracy is present even on this more persistent
subset, so treating all failures as short-lived label flicker is unsupported.
This does not identify why the model failed, validate the labels, justify using
future information in a model, or establish a general temporal capacity limit.
The descriptive aggregate is
`C:/CodexArtifacts/pretrained-reuse-r2/droid-persistent-change-error-20261028-attempt-01.json`,
SHA-256 `afa6632fcf515f2138a742c4197efb53e75063a7263ae05c150708871bdbfeba`.
No fitting, checkpoint selection, validation or sealed-audit data was used.

### Error type on persistent changes

A further post-hoc readback partitions the same 5,022 persistent rows into
correct new command, retained previous command, and a different wrong command.
All counts use the raw change-weighted probabilities on the same held train
folds. The linear control is the existing 50-epoch history-plus-sensor model;
its preprocessing and initialization differ from the LSTM, so this is not a
clean isolation of recurrence.

| Model | Correct new | Retains previous | Other wrong | Target NLL |
|---|---:|---:|---:|---:|
| LSTM 50 epochs | 128 | 4,712 | 182 | 2.1793 |
| LSTM 200 epochs | 238 | 4,424 | 360 | 2.1932 |
| Linear 50 epochs | 325 | 4,332 | 365 | 2.1336 |

Most wrong answers retain the previous command even when the new label persists.
Longer LSTM training increases correct changes, but also wrong non-previous
commands, and conditional target NLL does not improve. The seven target labels
occur 640, 654, 958, 575, 591, 760 and 844 times respectively; this is not a
single-class subset. These correlated observations are already exposed and do
not establish model generalization, causality, event noise or a capacity ceiling.
Future labels are used only to define this retrospective subgroup.

Aggregate output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-persistent-error-controls-20261028-attempt-01.json`,
SHA-256 `c6f841dd9b11a03895238b6b887367d9688aa118fc990cdeb3787cd996be2b8a`.
It pins the dataset summary and both saved prediction files; no new training,
checkpoint selection, validation or sealed audit was performed.
