# R2 LSC50 independent-group gate recheck (2026-11-14)

## Decision

Keep LSC50 `REVIEW_NOT_ADMITTED` for the current R2 multimodal-quality gate.
Do not acquire the 9.9 GB release, extract media, fit a readout, or evaluate
quality for this objective. The source is a plausible narrow sign-recognition
task with video and IMU, but it has only five participants. A participant-
disjoint train/calibration/development partition would have at most five
independent groups, far below the provisional 200-group threshold in the
draft R0 task-contract screen and too small to support a stable participant-
generalization or fusion-quality claim. This is a scope decision, not a claim
that the modalities cannot help.

## Evidence rechecked

- The pinned Figshare v1 record declares **CC BY 4.0** and currently offers a
  9.9 GB combined download. The already-preserved local audit has only bounded
  ZIP central-directory inventories and headers; no archive or payload was
  acquired for this recheck.
- The publisher's Data Descriptor reports 50 sign classes, 4,000 RGB/RGB-D
  videos and **five volunteers** (three native and two non-native signers).
  Video names encode sign, volunteer and repetition. IMU timestamps delimit
  signal intervals and the cameras were initialized together, so the source
  is plausibly synchronized at the recording level.
- The publication's participant count is the independent-group upper bound
  for a speaker-disjoint evaluation. Treating the many sign clips as
  independent would overstate generalization. The source is also a narrow
  Colombian Sign Language recognition task, not broad Omni decision evidence.
- Existing metadata inventory:
  `C:\CodexArtifacts\pretrained-reuse-r2\lsc50-source-audit-02`.
  Existing screen: [LSC50 readiness](R2M_LSC50_READINESS_20261012.md).

Sources:

- [Pinned Figshare v1 record](https://figshare.com/articles/dataset/LSC50_Colombian_Sign_Language_Video_and_Inertial_Measurement_Dataset/27383016/1)
- [Publisher's Data Descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11655960/)

## Scope and next action

This recheck read public source metadata and the tracked report only. No media,
participant records, timestamps, labels, model weights, features, or sealed
audit data were opened; no training, inference, or GPU work occurred. No code,
data, or prior experiment artifacts were changed.

Do not spend the local storage or runtime budget on LSC50 for the R2-M gate.
Continue with an already-screened candidate only if its independent group
count, rights, paired-input task definition and untouched development cohort
can be established from metadata before any payload access. Otherwise keep
R2-M open and report that no admitted source currently supports the required
fusion-quality evaluation.
