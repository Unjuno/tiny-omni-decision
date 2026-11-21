# Eidon Tracker POV metadata-only screen - 2026-11-22

## Decision

**Status: `NOT_ADMITTED_FOR_PRODUCT_QUALITY` and not yet admitted to the R2-M paired-input gate.** This is a plausible small video+temporal-state development source, not evidence for a four-modality model or a final quality benchmark. The entire public metadata index, including task-type labels and contributor IDs, was screened; this source must be treated as development-exposed. No blind audit set is claimed or created from it.

## Rights, content, and pinned revisions

The two publisher repositories declare CC BY 4.0. Their cards state that adult volunteers gave written consent covering research, commercial use, and redistribution. The same cards warn that footage is unprocessed, comes from private homes, and may incidentally include other people, documents, screens, or interiors; faces are not blurred. These facts make the source technically rights-cleared per its published terms but privacy-sensitive for model development and product use.

- Video + metadata repo: `eidon-ai/tracker-pov`, revision `47c55ccc6d0308894ab482009c6fa91569057a08`.
- Sensor repo: `eidon-ai/tracker-pov-imu`, revision `97d4441a7e915d2ad4b5988a43ebddd79a542aea`.
- Only the 1,121,123-byte `recordings/metadata.parquet` was downloaded, from the pinned video-repo revision. SHA-256: `d450018059bf6696ad793bab40fcb19fc49dcded8700df09047b2d508a8741e1`.
- Local file: `C:\CodexLocalArtifacts\pretrained-reuse-r2\eidon-tracker-pov-metadata-audit-20261122\metadata.parquet`.
- No MP4, IMU Parquet shard, sensor payload, user-demographics file, model, or checkpoint was downloaded or opened. No GPU was used.

## Metadata coverage and group feasibility

The index has 13,451 unique recordings across 27 contributor IDs. It lists video, task type, QC status, duration, sensor coverage, and file sizes. The contributor ID is the required split unit because it also identifies the same person, home, and camera rig.

The task labels are highly imbalanced: folding laundry 8,747 recordings / 860.2 hours; cleaning 2,384 / 276.3 h; dishes 984 / 52.1 h; cooking 795 / 39.5 h; and five tail categories with 19-166 recordings each. The largest contributor accounts for 16.09% of duration and the top five for 57.16%. The publisher reports that raw accelerometer/gyroscope/magnetometer values are opt-in and present in only 2,841 recordings (21.1%); orientation quaternions are more broadly available. Of the raw-motion recordings, 2,837 also have the complete seven-slot rig and chest reference. QC counts are 11,841 valid, 1,138 flagged, and 472 invalid.

A deterministic, exploratory five-fold contributor assignment was computed by sorting contributors by `SHA256("seed17:eidon-tracker-pov-contributor-fold-v1:<contributor_id>")` and distributing them round-robin. Fold group counts are 6/6/5/5/5, but example counts are 2,088/4,768/1,209/1,182/4,204. Among valid, full-rig raw-motion recordings, the five folds contain 404/695/72/48/1,116 examples. Several task classes are absent from some held-out folds; folds 2 and 3 have only 72 and 48 eligible examples. This deterministic diagnostic was not used to train or select a model, and is not an adequately supported nine-class generalization benchmark.

## Modality and task fit

The source provides egocentric video and seven body-mounted 9-axis IMUs at 24 Hz, with orientation data and (for a minority of records) raw motion values. The publisher says video and IMU were started back-to-back without hardware sync and should be aligned by timestamps; an offset of a few milliseconds is expected. The nine `task_type` labels form a coarse activity classification task, but their generation and human-verification procedure are not documented in the card. Nothing establishes that an activity decision needs both video and IMU; either modality may carry the coarse label on its own.

The source could support a narrow development diagnostic of temporal sensor/video fusion after an explicit task and privacy review. It cannot currently support a product-quality claim: there are only 27 independent contributors, activity labels are skewed, most recordings lack raw accelerometer/gyroscope/magnetometer data, label provenance is unspecified, and the full video corpus is 9.05 TB. Because the complete metadata and labels are now exposed, any future experiment on this source must remain development-only and use participant-grouped controls; do not call a partition from this index a blind sealed audit.

## Stop condition and re-entry criteria

No training or quality evaluation is justified from this metadata screen. Reconsider only for a narrowly scoped, explicitly non-product temporal-fusion pilot if a held-contributor protocol can retain adequate task and sensor support. The pilot must compare video-only, orientation-only, raw-motion-only where available, paired input, shuffled or time-offset sensors, and metadata-only priors on the same examples. It must report contributor-level variation and make no four-modality or sealed-audit claim. Do not fetch multi-gigabyte sensor shards or video until that task and privacy scope are fixed.

## Publisher sources

- [Video dataset card](https://huggingface.co/datasets/eidon-ai/tracker-pov/tree/47c55ccc6d0308894ab482009c6fa91569057a08)
- [Pinned metadata file](https://huggingface.co/datasets/eidon-ai/tracker-pov/blob/47c55ccc6d0308894ab482009c6fa91569057a08/recordings/metadata.parquet)
- [IMU dataset card](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu/tree/97d4441a7e915d2ad4b5988a43ebddd79a542aea)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
