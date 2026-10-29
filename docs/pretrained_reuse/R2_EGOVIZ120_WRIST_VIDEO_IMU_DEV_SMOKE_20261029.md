# EgoViz-120 development-group wrist video/IMU input smoke — 2026-10-29

## Decision

**Input-path smoke: PASS for this one record. Dataset admission: NOT PASSED.**
This verifies that one record from the frozen session-disjoint development side
contains decodable wrist video and finite wrist IMU data with close timestamps.
It is not a trained Decision result, a both-input quality result, or evidence of
participant-independent generalization.

## Frozen source and selection

- Dataset card: <https://huggingface.co/datasets/humaidtech/EgoViz-120>;
  the official source screen records the publisher's CC BY 4.0 statement and
  generated-label limitation in
  [`R2_EGOVIZ120_SOURCE_SCREEN_20261029.md`](R2_EGOVIZ120_SOURCE_SCREEN_20261029.md).
- Source index SHA-256:
  `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`.
- Split receipt SHA-256:
  `d2e2869dcce0a7d96cddc7dee639794a7cf253c3f922637ab9c9699cb5e6d7d2`.
- Split: seed 17, publisher `group_id`, deterministic SHA-256 order; 144/36
  train/development groups with no group or sequence overlap. Selected
  `chunk_id=7f2a80d98e54f0559893a5bf5cee0e5c`, `group_id=g008`, category
  `organizing_items`, sequence `s277`, duration 16 seconds, 480 indexed frames.
  The index and split are metadata-only; sealed-audit data were not created or
  accessed.
- The Hugging Face Bucket is mutable rather than revision-pinned. The bucket
  metadata helper returned 1,218 bytes for this path, inconsistent with the
  frozen index. The complete download was 153,903,851 bytes, matching the
  index, and its local SHA-256 is recorded below. This mismatch remains a
  source-metadata caveat; the local copy is content-pinned by its digest.

## Measured input path

One development MCAP was downloaded to a new external attempt directory. MCAP
CRC validation passed. The `/wrist_left/image/compressed` topic decoded as
`foxglove.CompressedVideo` protobuf carrying H.264: all 480 messages decoded
into 480 frames at 1920×1200. The `/wrist_left/imu/data` topic decoded as
`sensor_msgs/msg/Imu` CDR: all 3,200 samples had finite numeric fields. Every
video frame had a nearest IMU sample within 20 ms; offsets were median 1.281 ms,
p95 2.587 ms, maximum 3.481 ms.

The record is one session group only. The source still has 180 publisher
session groups, not verified participant identities; it remains below the
roadmap's provisional 200-independent-scene floor. Publisher-generated action
labels are not human-verified. This smoke did not read `/action` payloads or
train/evaluate a model. It therefore establishes transport/codec/time
compatibility only, not that an actual Decision target requires both inputs.

## Reproduction and artifacts

The smoke runner was executed twice on the same inputs; both JSON outputs were
byte-identical with SHA-256
`AA7F6C19FED03114656C5C0AEEBD00FF71FCCFFAAD121FC665325B9630CD3240`.
The repository runner SHA-256 is
`F4EB7E12EE606FC03C8A58702A5309C9A45FF8FC11ECC9CEA1DA111998809EB0`.

```powershell
$env:PYTHONPATH='C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-01\python-deps'
python scripts/audit_egoviz_wrist_video_imu_smoke.py `
  --index 'C:\CodexArtifacts\pretrained-reuse-r2\egoviz-source-screen-01\index.parquet' `
  --split 'C:\CodexArtifacts\pretrained-reuse-r2\egoviz-session-split-preflight-20261029-attempt-01\report.json' `
  --mcap 'C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-02\7f2a80d98e54f0559893a5bf5cee0e5c.mcap' `
  --output 'C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-02\wrist-video-imu-dev-smoke.json'
```

Environment: Python 3.11.9, mcap 1.6.0, zstandard 0.25.0, lz4 4.4.5,
PyAV 18.1.0, pandas 3.0.6, protobuf 7.36.1, huggingface_hub 1.29.0.
Ruff and `py_compile` passed. The receipt is checked into the accompanying
machine-readable manifest; the MCAP and all media remain outside Git at
`C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-02\`.

No GPU, model, training run, existing checkpoint, sealed audit, or legacy
worktree was modified. C: had 9.72 GiB free before this 153,903,851-byte
download and 9.58 GiB after it. Orbitune remained live throughout.

## Next gate

Do not fit a model from this one record. Reconsider EgoViz only if the source
can provide verified participant-independent groups and enough session groups,
and a preregistered paired task can show both-input necessity with matched
single-modality and mismatch controls. The current evidence justifies the next
bounded source/label audit, not architecture promotion or a product-quality
claim.
