# Native sensor execution smoke

The frozen training-only protocol completed on CPU float32 using source commit
`1afa8d10124eec83bbe3a240cb1f4eccf334e086`. Eight unique training windows were
consumed twice (16 encounters), with no held-out data read. The 6,700-parameter
stem received finite nonzero gradients and changed weights. Loss was 2.47260
then 2.45131. Total measured execution was 1.62 seconds, including saves and
payload-backed inference; this is not an isolated inference latency benchmark.

JSON probabilities were finite and normalized. Save/reload reproduced the exact
output; reversing options preserved every semantic probability and selection.
Initial and final weights remain separately under
`C:/CodexArtifacts/pretrained-reuse-r2/native-sensor-decision-smoke-01`.
The final PyTorch state file is 29,108 bytes; this excludes Python/runtime code.
[Exact hashes, environment and outputs](R2M_NATIVE_SENSOR_SMOKE_RESULTS_20261012.json).

This proves the bounded, text-free sensor execution path only. No Accuracy,
NLL/Brier/ECE on independent subjects, calibration, transition quality, fusion
benefit or general custom-option semantics has been established. All Omni quality
gates remain open. Next is a predefined subject-disjoint activity experiment
with a static control, temporal control and per-class probability metrics,
retaining mixed windows and the original frozen subject assignments. Calibration
and sealed-audit subjects remain untouched. Existing pretrained representations
remain the basis of the other modalities; this tiny fresh stem does not replace
them or establish a full Omni model.
