# SONIC-O1 audio-video source/license screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED` for this product-directed project. Do not download,
train on, or evaluate this dataset.** SONIC-O1 is unusually relevant to the
open R2 audio-video gate: it supplies synchronized natural conversation video
and audio, human-verified multiple-choice questions, temporal annotations, and
questions intended to use both modalities. However, the latest pinned dataset
README replaces an older CC BY 4.0 metadata declaration with a custom Vector
Institute License that limits access to non-commercial academic research or
Vector Institute sponsors/partners. It also requires branding for products
built using the dataset. No evidence shows this project qualifies for either
permitted category, and the goal is product-directed. The restrictive current
terms control this screen.

## Pinned source and task fit

- Dataset: [Vector Institute SONIC-O1](https://huggingface.co/datasets/vector-institute/sonic-o1).
- Dataset repository revision observed from the Hugging Face API:
  `71bce84b2686ac63693666966d592964e0690042` (last modified 2026-05-27).
- Pinned README:
  [README at revision `71bce84`](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/71bce84b2686ac63693666966d592964e0690042/README.md).
- Latest public code repository revision observed:
  `70180adaf179cf896b270a005f5c4aff08fe8c02`.
- The source reports 231 videos, about 60 hours, 1,335 five-choice MCQs and
  3,392 temporal-localization questions across 13 conversation domains. MCQs
  contain four content answers plus “Not enough evidence.” The paper describes
  the benchmark as native audio-video reasoning and says expert-verified
  rationales reference audiovisual evidence. This is a strong task fit in
  concept, not evidence that every item needs both modalities or that this
  project may use the data.
- The current Hub repository lists 885 files and about 56.4 GB of storage
  (`usedStorage` API field); the listed media portion is approximately 51.4 GB
  of video and 5.45 GB of audio. This makes a complete acquisition unnecessary
  even if rights were later resolved.

## Rights and version conflict

The current pinned README states that the dataset may only be accessed and
used by **academic entities for non-commercial academic research** or **Vector
Institute sponsors and partners**. It requires the notice “Built with Vector
Institute SONIC-O1” on products or services built using the dataset. The
README does not link a complete license text or establish that this project is
an eligible academic entity or sponsor.

An older Hub README revision (`08503fac147f3fb6c30d47b9283aacd5a3103384`)
was surfaced with `license: cc-by-4.0` metadata. The latest repository
revision’s README no longer has that license metadata and instead contains the
custom restrictive terms above. The paper describes the evaluation suite as
released under a “research license,” consistent with the current restriction.
The old CC BY display is not sufficient authorization to use the current
dataset release for this product-directed goal.

The dataset authors say their source YouTube videos were individually
published under CC BY 4.0. That source-level claim does not override the
current dataset-level access/use terms. The full custom license and eligibility
remain unverified.

## What was inspected

This was a metadata-only screen. It queried the dataset repository revision,
file tree, sizes, README, official project page, and paper. It did **not** load
question/answer JSON, demographic annotations, or media bytes. No dataset
payload, model, or checkpoint was downloaded; no features, split, evaluation,
or training run was created. No license was accepted, and no contact was sent.

## Decision

Keep SONIC-O1 unadmitted. Reconsider only if the rights holder supplies a
current written license that permits the intended product development and
distribution, and the project is independently confirmed eligible. If cleared,
freeze a video-grouped development split before inspecting targets, omit
demographic fields, and compare text-only, audio-only, video-only, and joint
probability readouts with video-level uncertainty and source-matched controls.
Do not treat any published benchmark score as a blind evaluation.

## Official references

- [Pinned SONIC-O1 dataset README](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/71bce84b2686ac63693666966d592964e0690042/README.md)
- [SONIC-O1 project page](https://vectorinstitute.github.io/sonic-o1/)
- [SONIC-O1 paper](https://arxiv.org/abs/2601.21666)
- [SONIC-O1 code repository](https://github.com/VectorInstitute/sonic-o1)
