# Humanola egocentric synchronized hand/IMU sample screen — 2026-10-29

## Disposition

`REVIEW_NOT_ADMITTED` for R2-M quality fitting. This is a metadata-only screen;
no video, Parquet, IMU, feature, or annotation payload was fetched, and no access
terms were accepted.

## Pinned public evidence

- Hugging Face dataset: [`humanola-inc/egocentric-sample`](https://huggingface.co/datasets/humanola-inc/egocentric-sample/tree/3c6e782911b4e82a56b288f6830d3a43bdb85d49), pinned repo revision `3c6e782911b4e82a56b288f6830d3a43bdb85d49`.
- Its card declares CC BY 4.0 and describes a 9-episode, 27-minute sample with 48,272 frames, four synchronized 1920×1200 camera streams, three approximately 200-Hz IMUs, and tracked 3D hand pose.
- The official [Humanola catalog](https://humanola.com/catalog) says catalog data is licensed per project and that every episode ships with provenance and recorded participant consent. The inspected material does not expose a consent/provenance manifest linking these claims to this exact HF revision, so retain a review flag rather than infer that linkage.
- The official card describes household recording; incidental faces, screens, documents, or interiors may be visible and are not blurred. The structured hand-pose target also makes this a narrow manipulation source, not general AV reasoning.

## Payload and split constraints

The unauthenticated public Hub tree metadata lists 99 files totaling
18,422,426,487 bytes; the largest file is 818,856,664 bytes. This is not an
estimate from the card's repository-size badge: it is the sum of per-file sizes
in the pinned tree metadata. Current C: free space was about 0.37 GiB, so even
the largest single video cannot be downloaded safely under the existing disk
reserve. No large transfer or disk cleanup was attempted.

The card lists only nine episodes and no participant-grouped train/validation
split or participant-to-episode manifest. Two episodes share the same listed
task, washing dishes. From this public evidence there is no defensible
independent participant split or adequate sample-size basis for a quality
claim. A future bounded runtime smoke would require a verified small asset,
participant-group provenance, and explicit linkage between the repo revision,
license, and recorded consent. It would not by itself establish multimodal
decision quality.

## Conclusion

This is not a practical or sufficiently independent R2-M dataset in the
current environment. Keep it unadmitted; do not infer task quality from the
presence of synchronized pose and IMU annotations. No model, source corpus,
checkpoint, or sealed audit was opened or changed.
