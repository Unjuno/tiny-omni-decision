# Hoi! force/video source screen — 2026-11-22

## Decision

Status: `REVIEW_NOT_ADMITTED`. Hoi! is a plausible source for the optional
sensor part of R2-M: the publisher describes synchronized video, force/torque,
and tactile streams across 3,048 interaction sequences, 381 articulated
objects, and 38 environments. The paper also confirms sequence-level
open/close state-change labels, articulation type/axis labels, and evaluation
tasks for articulation and force estimation. These are not already packaged
as supplied-option JSON decisions, but articulation type and state-change
labels could support explicit two-option decisions after the mapping is
audited. The publisher labels the dataset CC BY 4.0 and states commercial
adaptation is allowed with attribution. The exact per-row label schema, group
IDs and leakage-resistant split remain unverified.

Do not download dataset payload or create training examples yet. The publisher
dataset page indicates a guestbook asking for a name at download. No personal
information or access request has been submitted. Continue only after the
task/label structure and the permitted download flow can be checked without
submitting user information.

## Evidence reviewed

- On 2026-11-21 18:12:18 UTC, the official Dataverse file-list API returned
  4,769 metadata entries (4,258,029 response bytes; SHA-256
  `ee1b129467d2b68ebc32aafef099ba351406c7ddc0bb8ff36418a8e7406e4d51`). The
  metadata reports 3,186,202,640,224 total file bytes (~3.19 TB). Its entries
  have `restricted=false`, but all 4,769 set `dataFile.fileAccessRequest=true`;
  this is not evidence that payload downloads are guestbook-free. No file
  payload endpoint was requested and no name was submitted.
- The inventory contains 20 files named `manifest.csv`, while the paper
  reports 38 environments. The meaning and coverage of those manifests are
  not established, so they cannot yet be treated as the paper's environment
  split. The `bathroom_1/gripper` directory subtree alone lists 1,421 files
  totaling 1,274,184,763,539 bytes (~1.27 TB); this confirms that selective
  acquisition must be planned before any download.
- Official Bonn repository identifies the dataset as version 1.0 and CC BY
  4.0; it describes 3,048 sequences, 381 objects, 38 environments, four
  manipulation embodiments, and synchronized force/tactile for the instrumented
  gripper. Its file table lists 4,769 public files. It also reports a guestbook
  name field for file downloads.
- Official project page says synchronized force/torque and DIGIT tactile
  sensing are available with the Hoi! gripper and that the streams are
  temporally aligned. It explicitly says the CC BY 4.0 release can be adapted
  commercially with attribution.
- The official paper reports sequence-level open/close state-change labels,
  annotated articulation type/axis, and benchmarks for articulation type,
  tactile force estimation, and visual force estimation from RGB-D plus a
  manipulation goal. It reports seven demonstrators and alternating open/close
  interactions. This provides viable label concepts, but option sets and
  target extraction must be audited before use.
- The official code repository describes time alignment, interaction
  segmentation, and packaging steps, but this repository screen does not
  confirm row-level label schema, stable participant identity fields, or
  held-out group definitions.
- The dataset is too large to download as a single package. The listing shows
  one bathroom/gripper stream's depth archive at 5.5 GB and RGB archive at
  1.4 GB, which makes indiscriminate mirroring unsuitable for this C-drive
  constrained workflow.

## Admission gates still open

1. Inspect annotation schema and confirm a nontrivial option decision from
   articulation/state-change labels, without leaking labels through the
   selected video or force window. Prefer two-choice articulation or
   state-change decisions; do not threshold continuous force targets without a
   prespecified task rationale.
2. Establish split units and leakage checks across interaction, object,
   environment and demonstrator. The paper reports seven demonstrators, but
   accessible metadata has not confirmed stable identity fields or split
   membership.
3. Confirm which instrumented embodiments have enough examples
   for both visual-only and force/tactile-plus-visual comparisons.
4. Read the full dataset Readme and annotation payload only after the user has
   approved the access flow; the public file-list metadata is insufficient to
   establish the row schema or group splits.
5. Estimate storage using selected train/validation files before any payload
   acquisition; never mirror the full repository by default.

If admitted later, keep sealed-audit files isolated, group all views and
repeated sequences of the same object/environment as required, and evaluate
paired modality necessity with fixed labels and a pretrained visual encoder.
This screen does not admit the dataset into training or evaluation corpora and
does not establish model quality.

## Source links

- [Official Bonn dataset record](https://bonndata.uni-bonn.de/dataset.xhtml?persistentId=doi%3A10.60507%2FFK2%2FQODWTV&version=1.0)
- [Official project website](https://timengelbracht.de/Hoi-Dataset-Website/)
- [Official dataset tools repository](https://github.com/timengelbracht/hoi-dataset-tools)
- [Official paper](https://arxiv.org/abs/2512.04884)
