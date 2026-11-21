# Hoi! force/video source screen — 2026-11-22

## Decision

Status: `REVIEW_NOT_ADMITTED`. Hoi! is a plausible source for the optional
sensor part of R2-M: the publisher describes synchronized video, force/torque,
and tactile streams across 3,048 interaction sequences, 381 articulated
objects, and 38 environments. The publisher labels the dataset CC BY 4.0 and
states commercial adaptation is allowed with attribution. This is promising
for testing when contact sensing adds information to vision, but current public
metadata does not establish that it has the supplied-choice JSON decision
labels or unambiguous splits needed by this project.

Do not download dataset payload or create training examples yet. The publisher
dataset page indicates a guestbook asking for a name at download. No personal
information or access request has been submitted. Continue only after the
task/label structure and the permitted download flow can be checked without
submitting user information.

## Evidence reviewed

- Official Bonn repository identifies the dataset as version 1.0 and CC BY
  4.0; it describes 3,048 sequences, 381 objects, 38 environments, four
  manipulation embodiments, and synchronized force/tactile for the instrumented
  gripper. Its file table lists 4,769 public files. It also reports a guestbook
  name field for file downloads.
- Official project page says synchronized force/torque and DIGIT tactile
  sensing are available with the Hoi! gripper and that the streams are
  temporally aligned. It explicitly says the CC BY 4.0 release can be adapted
  commercially with attribution.
- The official code repository describes time alignment, interaction
  segmentation, and packaging steps, but this repository screen does not
  confirm per-subject labels, held-out group definitions, or multiple-choice
  decision targets.
- The dataset is too large to download as a single package. The listing shows
  one bathroom/gripper stream's depth archive at 5.5 GB and RGB archive at
  1.4 GB, which makes indiscriminate mirroring unsuitable for this C-drive
  constrained workflow.

## Admission gates still open

1. Verify available labels can support a nontrivial supplied-choice decision,
   rather than simply reconstructing an action annotation.
2. Establish split units and leakage checks across interaction, object,
   environment and any participant identity. The metadata screen does not
   establish those identifiers.
3. Confirm which instrumented embodiments have enough independent examples
   for both visual-only and force/tactile-plus-visual comparisons.
4. Read the full dataset Readme and file metadata through an access path that
   does not submit personal information, or wait for the user's own download.
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
