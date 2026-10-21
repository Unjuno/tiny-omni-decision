# AVQA source screen — 2026-10-21

## Decision

AVQA is a strong **task-design candidate** for cross-modal decision work: its
official page describes hand-annotated multiple-choice questions with a
`question_relation` field (`View`, `Sound`, or `Both`). The page explicitly
describes questions whose answer requires relating audio and visual events.
However, **do not admit AVQA to training or evaluation**. The available source
terms do not establish the rights needed for model-development use of the
annotations and source clips, and the source video rights remain tied to the
original owners. No annotations, features, or media were downloaded or opened.

## Pinned public evidence

This is a dated web-metadata screen; the publisher does not expose an immutable
dataset revision on the public project page. Exact HTML snapshots were fetched
to `C:/CodexArtifacts/pretrained-reuse-r2/avqa-source-screen-20261021-01/`:
AVQA page SHA-256
`ba761adbdef59e4c644c053e531dc9d5ebf7ec615216b8e6e56561cb6e2dd8b9`, and
VGG-Sound page SHA-256
`33d2c7dd5d8dcc878d772f485380cbe2b56b04fd54be9929612dfd892c602e8e`. The
manifest's required 40-hex `revision` field contains the AVQA page snapshot
SHA-1 only; upstream dataset revision remains `UNKNOWN`.

- [Official AVQA project page](https://mn.cs.tsinghua.edu.cn/avqa/) says the
  videos come from VGG-Sound and points to YouTube IDs, timestamps, and
  train/test split metadata. It describes manually annotated multiple-choice
  questions and labels the modality relation for each question. Its license
  section permits personal/classroom copies under conditions, says component
  owners' copyrights must be honored, and requires specific permission or a fee
  for redistribution or server posting. That notice does not clearly grant
  machine-learning training rights for the annotations or original clips.
- The linked [official VGG-Sound page](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
  says the underlying clips are from YouTube, the original video owners retain
  copyright, and its prior dataset download links are no longer available.
  Its CC BY 4.0 statement for VGG-Sound does not, by itself, resolve each
  original clip's rights or expand AVQA's separate annotation terms.
- AVQA's download link redirected to a Microsoft sign-in page during this
  screen. No access was requested or accepted.

## Task and split risks

The `Both` relation makes AVQA more relevant than generic audiovisual event
classification. Still, the available page alone cannot verify the number of
usable `Both` examples, whether those questions truly require both channels,
or whether the supplied train/test assignment groups all questions by the
underlying YouTube source video. Its displayed `val_qa.json` is described as
the test annotations; it must remain closed during development. Do not use
reported AVQA scores as Tiny Omni Decision results.

Reconsider only after an authorized annotation release and explicit source
media rights are established, then audit source-video grouping and define
train/validation without reading the published test annotations. Keep a
question-only and unimodal control, and require truth-bearing input ablations
before claiming complementarity. The screen did not inspect candidate labels,
download test files, load models, or run training.

## Disposition

`manifests/candidates/avqa.yaml` records the source as `REVIEW`; the repository
manifest audit returns `project_policy: DENY` for evaluation because the
annotation component does not establish the required use rights. It is not
included in a training or evaluation catalog. This source screen narrows the
candidate set but does not close R2 M-2 or the four-modality quality objective.

## Verification

- `validate-dataset-manifest manifests/candidates/avqa.yaml`: PASS; reports
  `project_policy: DENY`.
- Every candidate dataset manifest and the training/evaluation catalogs:
  PASS.
- CPU suite: **2,305 passed, 11 skipped**; Ruff: PASS; `git diff --check`:
  PASS.
- No model was loaded and no optimizer updates occurred.
