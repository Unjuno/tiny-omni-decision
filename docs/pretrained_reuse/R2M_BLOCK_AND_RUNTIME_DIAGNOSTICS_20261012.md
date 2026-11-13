# VIDIMU input/gradient and delayed sensor runtime diagnostics

These diagnostics do not fit or select a new model. Existing weights, data and
validation conditions remain unchanged. Sealed audit payloads are unopened.

## Training-only block audit

At the saved joint training normalization, video has6144 nonconstant dimensions
and sensor360. Their training mean squared block norms are6144 and360: ratio
17.0667. Per-coordinate standardization does not equalize whole-block energy.
This geometric fact is not causal proof of the fusion quality failure.
Evidence: `vidimu-training-block-audit-01/result.json`.

A full-training-batch backward pass was then measured without any optimizer
update at seed17 initialization and the saved selected checkpoint. State tensors
were checked unchanged after backward. Training CE:2.806624 initial,0.110283
selected.

| State | Block | Gradient L2 | Gradient RMS | Logit contribution RMS |
|---|---|---:|---:|---:|
| initial | video |16.787830|0.059402|0.567451|
| initial | sensor |4.685821|0.068496|0.138527|
| selected | video |1.032705|0.003654|7.638342|
| selected | sensor |0.311120|0.004548|0.586553|

Sensor gradients reach the head and their per-coordinate RMS exceeds video's;
the evidence does not support a disconnected sensor-gradient path. Logit block
RMS includes class-common components and is not a calibrated contribution or
causal attribution. AdamW's adaptive updates also mean raw gradient norms alone
do not establish optimization dominance. No scaling ratio was tuned against
validation. Evidence: `vidimu-training-gradient-audit-01/result.json`.

## Real delayed-window JSON boundary

The first fixed HARTH validation window was replayed through the saved static
model after verifying subject-data and checkpoint hashes. Availability was set
to the complete128-sample window end. At the center target timestamp the wrapper
abstained; at the end it invoked the real model once and returned JSON;21ms
later it abstained under the explicit20ms diagnostic age limit. That limit is a
boundary test, not an optimized product freshness policy. This window's
end-minus-target delay is1260ms.

Evidence: `harth-delayed-json-replay-01/result.json`, including source/weight
hashes and full output. This is one input-path test, not a new quality evaluation,
streaming-throughput measurement, or claim that the model predicts current-time
labels. The adapter used synthetic metadata channel names; production must bind
actual channel names/calibration and payload verification, not copy that fixture.

All artifact directories are under `C:/CodexArtifacts/pretrained-reuse-r2`.
Remaining product gaps include useful fusion, arbitrary criteria, quality across
all required modalities, calibrated abstention, real streaming adapters and
end-to-end deployment measurements. These diagnostics do not close those gates.
