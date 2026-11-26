# Center readout comparison: not adopted

The fixed12epoch run completed from71d6eb0; minimum validation NLL selected
epoch11. All62,269 training windows and12,763 validation windows were retained.
The selected6,700-parameter center readout scored Accuracy89.2502%, NLL0.309974,
versus mean91.4675%/0.241235 and static93.8729%/0.189363. Best reload was exact.
The architecture and initialization were parameter-identical to temporal mean;
only the pooling location changed. No quality gate passed.

The result does not support replacing global mean with a single local center
feature at this budget. A9sample center receptive field discards much temporal
context. Do not infer that all center-aware or temporal representations fail.
Reject this candidate; stop the bounded sensor readout series as preregistered,
returning to missing native screen and genuine crossmodal tasks. No extra epochs,
seeds, compression or product adoption follow from this result.

Artifacts remain separately at
`C:/CodexArtifacts/pretrained-reuse-r2/sensor-center-training-01`, with initial
epoch evaluation, all curves/predictions, selected/final checkpoints and config.
[Full probability metrics, time and hashes](R2M_SENSOR_CENTER_RESULTS_20261012.json).
Calibration and sealed audit remain unopened. This is another comparison on
development validation, not new blind evidence. Existing models remain intact.
