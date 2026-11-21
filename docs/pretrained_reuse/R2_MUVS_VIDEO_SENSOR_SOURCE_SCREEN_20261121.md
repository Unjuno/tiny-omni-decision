# MUVS video + device-sensor source screen — 2026-11-21

## Decision

`METADATA_TRAIN_PREFLIGHT_COMPLETE_VIDEO_NOT_ADMITTED`: the Zenodo-deposited
metadata/annotation archive is identified as CC BY 4.0 and has now been
downloaded and audited using publisher-train rows only. The separately hosted
GCS video remains outside the admitted scope because its license is still
`TO BE ADDED`; do not access or use that media until rights are clarified.
This audit establishes neither video+sensor task quality nor product
suitability.

## Rights and publisher facts

- Zenodo record `20708683`, v1.0, DOI `10.5281/zenodo.20708683`, lists the
  deposited record as CC BY 4.0. The downloaded `data.zip` is 9,742,259 bytes,
  MD5 `8fd1d426e0b4308655ea71884d1b0c8e`, SHA-256
  `df25517793df22f9f57eef1f8f0ffba131d98b5958313110a5d9a7592820912b`.
- The README says video is hosted separately in GCS and its license is
  `[TO BE ADDED]`. The official project site reserves copyright. No GCS video
  was accessed or downloaded; the deposit license is not assumed to grant
  rights to those videos.
- The README claims 11 train and 6 test events. The archive instead contains
  fragment filenames for 12 train and 5 test events, with no event-name
  overlap. This discrepancy is unresolved.

## Train-only metadata preflight

The audit opened only train fragment and matching annotation CSV members. It
listed test fragment filenames from the ZIP directory for split accounting,
but did not decompress or parse test fragment/annotation member contents or
test labels/sensor values. The ZIP as a whole was downloaded and hashed, so
the test member bytes were transferred as part of the archive; they were not
opened by the audit.

- 3,208 labeled train rows across 12 fragment events; 3,181 have at least two
  available camera choices and are eligible for a decision task. The other 27
  have only one choice and are excluded from that interpretation.
- Choice counts: 1 option 27, 2 options 1,063, 3 options 2,118. Every selected
  camera was present among the corresponding candidate choices.
- A camera-ID-frequency baseline trained on the other 11 events and evaluated
  leave-one-event-out scores 54.86% over the 3,181 multi-option rows, against
  a 38.90% uniform-choice expected accuracy. This is a metadata-only shortcut
  baseline, not a learned video/sensor result; it reveals event-level label
  prior variation and raises shortcut/generalization concerns.
- `ActiveCameras` disagreed with the number of distinct camera IDs in 57
  annotated fragment groups. The meaning of this field and whether it should
  define choices need clarification before task construction.
- Six accelerometer/gyroscope axes each have 58 missing values out of 128,193
  train fragment rows (0.045%). One event has 195 fragment groups without an
  annotation among 290 available groups; the other events have no such gaps.
- The inspected README identifies one expert annotator; annotator agreement
  is not reported.

These properties make the annotation a plausible task proposal, but are not
evidence that it is a stable or useful task. No model was fitted, no video
quality was assessed, and no validation/test labels were used.

## Reproducibility and artifacts

- Candidate screen originally introduced at `4ba5313eda00fc2b4b7bb85b19fa9f7e4c0d94e2`; audit code was run from base commit `d64a7a910f979730fa8c68d49e1a17a0caeb2777`.
- Audit implementation: `src/tiny_omni_decision/muvs_preflight.py` and
  `scripts/audit_muvs_train_metadata.py`.
- Archive: `C:\CodexArtifacts\pretrained-reuse-r2\muvs-metadata-preflight-20261121-attempt-01\data.zip`.
- Refined report:
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\muvs-metadata-preflight-20261121-attempt-01\train-audit-02\train-metadata-preflight.json`.
- Versioned compact audit receipt: `docs/pretrained_reuse/MUVS_TRAIN_METADATA_PREFLIGHT_20261121.json` (24,862 bytes).
- The refined audit records train payload counts and hashes, confirms no
  train/test event-name overlap, and explicitly records test member contents
  as not decompressed or parsed.
- No training, GPU work, or cloud cost.

## Re-entry conditions

Before any video access, obtain written clarification that the separately
hosted GCS media permits the intended product-directed local model
development. Before model fitting, also resolve the archive/README split
count discrepancy, define candidate choices consistently despite the
`ActiveCameras` mismatches, and specify an event-grouped train/validation
protocol that does not inspect publisher test contents. Then compare
video-only, sensor-only, and paired systems with input shuffling, missing
input, and temporal-offset controls. Treat the 54.86% prior baseline as a
required shortcut baseline, not as model quality.

## Evidence

- Zenodo record and file listing: <https://zenodo.org/records/20708683>
- Read-only Zenodo API metadata: <https://zenodo.org/api/records/20708683>
- Publisher README preview: <https://zenodo.org/records/20708683/preview/README.md?include_deleted=0>
- Official dataset site: <https://swperfi-project.github.io/Pages-dev/MuvsDataset-page/>
