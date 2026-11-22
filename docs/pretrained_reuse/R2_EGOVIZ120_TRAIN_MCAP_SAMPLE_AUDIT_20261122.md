# EgoViz-120 train-only MCAP sample audit - 2026-11-22

## Decision

**Status remains `REVIEW_NOT_ADMITTED`.** This is a one-file, train-only media-integrity and synchronization audit. It shows that the selected raw H.264 stream is decodable and that its annotations broadly describe the visible activity. It does not resolve label verification, participant independence, or whether a Decision task genuinely requires both video and IMU. No model was trained or evaluated.

## Scope and immutable source identity

The audit uses the existing metadata index and its already-frozen session-group split; it creates no new split and reads no development, test, or sealed-audit payload. The metadata index has SHA-256 `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074` (870,961 bytes). The inspected file is the smallest MCAP in that index and belongs to frozen train group `g114`:

- Chunk: `c59801fe4172cf9b430cd9232dda3909`
- Category/title: `cleaning_restroom` / "Wiping bathroom sinks and faucets"
- Indexed duration: 15 seconds; 2 actions and 5 action phases
- Size: 150,184,348 bytes
- SHA-256: `c86d1d69df71953b0b7fe1d61cfbc919743d1c108a2a0adc30b5c1a449e2e7d4`

The working copy and decoded preview are under the actual C: directory `C:\CodexLocalArtifacts\pretrained-reuse-r2\egoviz120-train-mcap-audit-20261122-attempt-01\`. `C:\CodexLocalArtifacts` is not a junction. The older `C:\CodexArtifacts` path is a junction to D: and was not used as the final working destination. At the last capacity check C: had 205.44 GiB free. No dataset media or preview is in the repository.

## Decode and timestamp checks

A dedicated Python 3.11.9 virtual environment was used with `mcap==1.6.0`, `mcap-protobuf-support==0.6.0`, `mcap-ros2-support==0.6.0`, `av==15.1.0`, and `Pillow==11.1.0`; these are audit-only and were not added to project requirements. All decoding and inspection ran on CPU. The independent Orbitune GPU job was left untouched.

The head-left H.264 topic `/ego/zed_head/left/image/compressed` decoded all **449 of 449** frames at 1920x1200, with zero frames marked corrupt. The MCAP also contains 449 frames each for head-left and head-right, and 450 each for left and right wrist cameras. Head-camera timestamps match exactly; every head-left frame has a wrist frame within 5 ms. The three IMU streams contain about 2,992-3,001 records each (about 202 Hz); the nearest IMU samples are within 3.8 ms of head-left video frames. All 449 action messages align to head-left timestamps. The index and sample hashes, channel counts, and timing summaries are in the machine-readable [receipt](../../manifests/pretrained_reuse/egoviz120-train-mcap-sample-audit-20261122.json).

A single 640x400 local preview of frame 225 shows a first-person view of a sink-cleaning action, consistent with the broad clip label. The three sampled frames have non-uniform luma and distinct luma hashes. This visual spot-check is not a systematic label audit. The local preview SHA-256 is `e8e3dc5ec4b3d9c816f00673f9f25f2ca8dcafdb58fb8a39ae2ca5c83fe0ec4b`.

The five action-phase labels change at message indices `0, 41, 71, 285, 330`, while their declared phase starts are `0, 40, 70, 284, 329`. This is a one-message offset after the initial phase in this one clip. It may reflect a frame-index/end-point convention; it is recorded as an ambiguity, not asserted to be a labeling error and not corrected. The publisher describes the labels as model-generated and not human-verified.

## Limits and next gate

The sample confirms a technically usable, synchronized video+IMU stream for one train clip. It does not establish cross-session participant independence, a verified label target, a both-input-required task, or generalization. EgoViz-120 remains **not admitted for training or quality claims**. No split, model, checkpoint, or prior evaluation result was changed; no development/test payload or sealed audit was accessed.

Before fitting, obtain evidence about dataset-specific label review and participant identities across session groups. Then define a pre-registered choice task whose target cannot be recovered from metadata or either modality alone, keep every session/participant and parent sequence in one split, and compare video-only, IMU-only, paired, shuffled-IMU, and time-offset controls on the same train/validation examples. Do not claim paired-input benefit unless those controls support it. The publisher's stated full dataset size remains far beyond this sample audit; the availability of one small clip does not make full-corpus acquisition feasible.

## Related records

- [Original metadata screen](R2_EGOVIZ120_SOURCE_SCREEN_20261029.md)
- [License and label-provenance recheck](R2_EGOVIZ120_PROVENANCE_RECHECK_20261029.md)
- [Frozen session split preflight](R2_EGOVIZ120_GROUP_SPLIT_PREFLIGHT_20261029.md)
- [HABIT video+sensor integration gates](R2M_VIDEO_SENSOR_DECISION_INTEGRATION_20261121.md)
