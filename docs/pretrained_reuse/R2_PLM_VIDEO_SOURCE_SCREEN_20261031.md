# PLM Video Human/Bench source screen — 2026-10-31

## Disposition

Keep Meta's PLM Video Human/Bench **unadmitted**. Do not acquire data, train,
or use it to close R2-M. The follow-up inspected the official Meta dataset
page, the pinned Hugging Face dataset metadata/card/license, and the official
`facebookresearch/perception_models` repository. The Hugging Face dataset
viewer exposed a small preview of train annotations; no dataset files, video,
audio, or model weights were downloaded or opened.

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

## Pinned Hugging Face metadata follow-up

The official Meta download button resolves to
[`facebook/PLM-Video-Human`](https://huggingface.co/datasets/facebook/PLM-Video-Human).
Its Hub repository revision at this screen is
`9af8490af2630b87769a2b040c6c5b03c7752164` (last modified 2025-05-21).
The pinned `README.md` reports an FGQA train config of 2,321,035 rows and
409,709,782 annotation bytes. The Hub page and repository `LICENSE` declare
CC BY 4.0, while Meta's current PLM page explicitly places FGQA under the
Llama 3.2 license and imposes a `Llama` model-name condition. This is an
unresolved license conflict; do not use FGQA unless the provider clarifies
which terms govern this exact release and the project accepts them.

The repository contains annotation Parquet files and `metadata_train.jsonl`,
not source videos or audio. The FGQA schema records `source_video_id`,
`source_dataset`, and source time bounds. The public train preview identifies
upstream sources including COIN, HowTo100M, and Ego4D. The Hub config exposes
only a train split for FGQA; it does not provide a validation split or a
group-split manifest. The metadata may support a future source-ID audit, but
does not grant rights to the referenced media or establish participant/group
independence. No annotation Parquet or referenced video was downloaded.

## Next gate

Before any access, a future screen must verify the exact task-specific files,
source-asset licenses, split/group IDs, and media size from official metadata.
If rights and split integrity pass, first consider a bounded train/validation
video-only M1 probe on a task with supplied options or unambiguous
programmatically checked options. Reserve an independent group-level
validation source; the FGQA train config alone cannot supply one. Do not use
FGQA pending resolution of the license conflict. This candidate does not
justify fusion training or encoder unfreeze.

## Primary sources

- [Meta PLM Data page](https://ai.meta.com/datasets/plm-data/)
- [Pinned PLM Video Human Hub dataset card](https://huggingface.co/datasets/facebook/PLM-Video-Human/tree/9af8490af2630b87769a2b040c6c5b03c7752164)
- [Pinned Hub README](https://huggingface.co/datasets/facebook/PLM-Video-Human/blob/9af8490af2630b87769a2b040c6c5b03c7752164/README.md)
- [Official Perception Models repository](https://github.com/facebookresearch/perception_models)
- [Official PLM license file](https://github.com/facebookresearch/perception_models/blob/main/LICENSE.PLM)

