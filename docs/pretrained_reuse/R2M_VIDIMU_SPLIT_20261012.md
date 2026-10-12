# VIDIMU subject split frozen before payload inspection

The checksum-verified original `videoandimus` inventory contains 208 raw IMU
trials: 16 subjects, each with all 13 activity IDs. Corrected copies are not
independent examples. No sensor arrays, video frames or per-example quality
results informed this split.

Sorted subject IDs are shuffled using Python Random(17), then assigned once:

- Train (130 trials): S48, S52, S56, S51, S44, S49, S40, S41, S53, S50.
- Development validation (39 trials): S42, S57, S47.
- Sealed audit (39 trials): S46, S54, S55. Payloads remain unopened.

All windows, views and corrected variants from one subject inherit its group.
No calibration subset has yet been allocated; validation probabilities must not
be described as independently calibrated. No subject reassignment is permitted
because of downstream performance or synchronization problems. Any unavailable
trial must be recorded explicitly; no replacement from another split.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-frozen-split-01/split.json`.
SHA256: `230e4f9f9fb6218b90fea0894396bc96bf67a0951864301d0fe289f4a0457b2e`.
The artifact pins archive and inventory hashes and records synchronization as
unverified. `scripts/freeze_vidimu_split.py` verifies the archive and exact
subject/activity coverage, and refuses to overwrite an existing output.
Two CPU tests cover order independence, subject disjointness and malformed
coverage. This is split preparation, not approval of temporal alignment or
evidence of multimodal quality.
