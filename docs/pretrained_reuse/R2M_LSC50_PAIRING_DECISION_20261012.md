# LSC50 synchronized pairing: not established, stop acquisition

Inspected complete upstream `LoadFiles.ipynb` at commit
`4d35fba04786e4b6d3654894a222b9fa5662a3ad` without executing it.
SHA256 `4a24caa34c5b2ffbea188a746c15ab160932a834929bc63431aacf069e1b9c5a`.
It loads landmark CSVs and a user-specified video via OpenCV. It contains no IMU
interval loader, timestamp transform, volunteer-index mapping or repeat matching.
Thus the earlier250IMU-versus1000video-trial ambiguity remains unresolved.

Decision: stop LSC50 media/sensor acquisition for the synchronized-fusion branch.
The dataset is not declared defective; its published information inspected here
does not prove the correspondence our experiment requires. License remains
CC-BY4.0. Do not fabricate pairs or claim synthetic pairings are contemporaneous.
Preserve inventories and timestamp metadata for future clarification. No weights,
raw sensor data or video recordings were downloaded; no training performed.

Source snapshot: `C:/CodexArtifacts/pretrained-reuse-r2/lsc50-loader-audit-01`.
This is a source-specific exclusion, not a whole-goal block. Next candidate is
VIDIMU, whose descriptor explicitly discusses stored synchronization corrections.
