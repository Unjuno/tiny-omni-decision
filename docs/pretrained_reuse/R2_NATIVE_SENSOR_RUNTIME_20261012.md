# Native sensor JSON and time semantics

The selected 2,380-parameter static HARTH head now has an explicit window
inference entry point. It requires all twelve activity codes, calibrated six-axis
acceleration in g, 128 samples at 50 Hz, and a query on the observation clock
at or after the final evidence sample. Missing samples return abstention.
Legacy subset inference remains unchanged.

The predicted label belongs to sample 64, not the query instant. JSON includes
evidence start/end, label time and the 1,260 ms of future evidence used relative
to that label. Evidence end is not a measured compute/transport completion time.
This is delayed central-label recognition, not causal recognition at label time.

## Actual replay

Command: `python -m scripts.replay_native_sensor_json --output C:/CodexArtifacts/pretrained-reuse-r2/native-sensor-json-replay-01`

Eight existing validation windows, CPU inference, no training or new quality
evaluation. Input-only NPZ contains values/timestamps, without labels. All twelve
probabilities are preserved in custom-ID JSON. Compared with saved best-epoch
predictions, maximum probability difference is 1.589394647005804e-7, below the
predeclared 1e-5 gate; zero answers change. Source and checkpoint hashes remain
unchanged. Full decisions and provenance are in the output's result.json.

Selected weight SHA-256:
`5780b04c5022de2f6a085c27032de05c5f86e84a6c638b02a93ed91a9b7db40d`.
Reference prediction SHA-256:
`4d36521cf665c28deeaaeb0329a6cff89d4dc53f7a7490d0e4a22a0577d3029c`.

Tests cover timing rejection, mismatched clocks, nonfinite query time,
full-taxonomy enforcement, option reordering and missing-input abstention.
This does not establish general sensor semantics, joint fusion quality or the
95–98% product target. No audit data, cloud compute or artifact overwrites.
