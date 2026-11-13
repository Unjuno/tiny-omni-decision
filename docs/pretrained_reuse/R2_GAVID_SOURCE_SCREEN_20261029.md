# GAViD source screen — 2026-10-29

## Decision

`NOT_ADMITTED_PRODUCT`. GAViD is relevant as a synchronized audio/video group-
affect benchmark, but the publisher describes the dataset as academic research
only and non-commercial. The product-directed workstream must not rely on it
without clarified compatible rights. No archive, media, or dataset CSV was
downloaded.

## Evidence and fit

The [Zenodo record](https://zenodo.org/records/15448846) reports 5,091 five-
second clips with video, aligned audio, generated context, and group-affect,
emotion, intensity, interaction, and action-cue labels. It reports 3,503 train,
542 validation, and 1,046 test samples, while stating that only training clips
are currently provided and validation/test clips require a request and
approval. Although the publisher says the source YouTube videos were CC BY 4.0,
the dataset record separately states non-commercial academic-use terms and
identifies a custom dataset license. The source-video claim does not override
those dataset terms.

The [paper](https://arxiv.org/abs/2604.16214) frames the task as context-aware
group affect recognition. That is not yet evidence for option-conditioned
Decision behavior or for both audio and video being necessary. In particular,
generated video context could act as a shortcut. A research-only study would
need to compare audio-only, video-only, context-only, and joint conditions and
test modality corruption or replacement on a split grouped by original source
video. No such experiment is justified under the current product-use gate.

## Reproducibility and disposition

Metadata-only review of the [publisher record](https://zenodo.org/records/15448846)
and [paper abstract](https://arxiv.org/abs/2604.16214) on 2026-10-29. No
repository checkout, CSV, archive, media, training, or sealed-audit access.
Machine-readable record: [GAViD source screen](../../manifests/pretrained_reuse/gavid-source-screen-20261029.json).
