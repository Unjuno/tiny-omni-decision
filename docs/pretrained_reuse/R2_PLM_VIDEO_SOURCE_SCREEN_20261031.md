# PLM Video Human/Bench source screen — 2026-10-31

## Disposition

Keep Meta's PLM Video Human/Bench as a **video-only M1 candidate under review**.
Do not acquire data, train, or use it to close R2-M. This is a metadata and
license screen of the official Meta dataset page and official
`facebookresearch/perception_models` repository; no dataset files, annotations,
media, or model weights were downloaded or opened.

## What is promising

The official page describes several decision-adjacent temporal tasks:

- Fine-grained question answering (FGQA), with MCQ evaluation options that
  differ in details such as action direction.
- Region Temporal Localization (RTLoc), which selects a time interval given a
  video, region masklet, and event description.
- Region Video Captioning and Dense Temporal Captioning, which test
  temporally grounded event understanding.

These could provide a more natural video-language task than CLEVRER. The
official page also documents a separate smart-glasses QA evaluation component,
but it does not establish the input schema, task-level media access, or
whether its questions can be converted to supplied-option decisions without
introducing new labels.

## Rights and data-integrity gates

- Meta states that PLM Video Human/Bench data is CC BY 4.0, **except FGQA**,
  which is an output from Llama 3.2 and subject to the Llama 3.2 license. Meta
  further states that a distributed/improved model trained on that data must
  have a name beginning with “Llama”. Do not include FGQA unless that model
  naming/license condition is explicitly acceptable for this project.
- The page says FGQA questions concern clips from open-access video datasets,
  but does not provide per-asset source licenses or prove that each underlying
  video grants the rights required for product-directed training. A dataset
  annotation license alone does not resolve underlying video rights.
- The official page does not establish, in the reviewed metadata, a complete
  per-task video/group manifest, exact train/validation group disjointness, or
  whether a worker/source identity can span groups.
- The reported task inputs are video, text, masks, and temporal annotations.
  The page does not establish an audio-dependent decision task or demonstrate
  that two streams are jointly necessary. It therefore cannot close M-2.
- The repository describes large full-video datasets, but no size-bounded
  task-specific download plan or local-storage feasibility was verified.

## Next gate

Before any access, a future screen must verify the exact task-specific files,
source-asset licenses, split/group IDs, and media size from official metadata.
If rights and split integrity pass, first consider a bounded train/validation
video-only M1 probe on a task with supplied options or unambiguous
programmatically checked options. Keep FGQA excluded unless its license
condition is accepted. Reserve the official test split; do not use it for
selection. This candidate does not justify fusion training or encoder unfreeze.

## Primary sources

- [Meta PLM Data page](https://ai.meta.com/datasets/plm-data/)
- [Official Perception Models repository](https://github.com/facebookresearch/perception_models)
- [Official PLM license file](https://github.com/facebookresearch/perception_models/blob/main/LICENSE.PLM)

