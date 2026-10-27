# DROID dense train windows

The [fixed construction](R2_DROID_DENSE_TRAIN_WINDOWS_PROTOCOL_20261027.md)
created **80,998** windows from the 275 existing train recordings. Each target
step `t >= 8` has the sensor sequence `[t-8,t)`, previous command at `t-1`,
target at `t`, a recording index, target index and one of the original five
recording-group folds. There are **7,442** changed-command windows and 73,556
stable-command windows. The full 7×7 previous/target count matrix exactly
matched the earlier train-recording census.

The fold recording counts are 58, 57, 56, 53 and 51. Windows in one recording
remain in the same fold; no rebalancing, repeated rows, validation labels,
image decoding or final evaluation was performed. Nonselected opaque TFRecord
records were read only for identity checks. Each prior fixed episode target,
source shard length and hash, and all 275 recording identities matched.

Train windows are correlated, and these folds do not establish independence
across operators or locations. The dominant-axis change definition may react
to small command fluctuations; raw coordinate increments are not physical
velocity estimates. These artifacts do not establish multimodal quality.

External output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-dense-train-windows-20261027-attempt-01/`.
The five NumPy arrays total about 37 MB. Raw SHA-256 of `summary.json`:
`31abc71b9d31f69e906c6e98ee712c82384dc90bbd5ab1a9a6122b0c12953982`.
Its receipt pins every array, private recording/fold index, source, protocol
and code hash. CPU construction and source checks took 27.70 seconds.
