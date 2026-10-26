# AVMIT natural audio/video source screen — 2026-10-26

## Scope

Read-only admission screen for Audiovisual Moments in Time (AVMIT) as a
possible synchronized AV Decision source. No AVMIT metadata files, embeddings,
video or audio payloads were downloaded or opened. No model inference, fitting,
test-set inspection, or access request was performed.

Pinned source checked: the official repository
[`mjoannou/audiovisual-moments-in-time@e3c06ec0e215b1934d141f86008e92e567366d6c`](https://github.com/mjoannou/audiovisual-moments-in-time/tree/e3c06ec0e215b1934d141f86008e92e567366d6c).
Its repository license is CC0-1.0 (GitHub API license blob
`0e259d42c996742e9e3cba14c677129b2c1b6311`). This records the repository
license; it does not by itself resolve the rights in source video/audio or
separately deposited embeddings.

## Task and grouping potential

AVMIT records three human ratings for 57,177 three-second videos. The raters
were asked whether a supplied action label was present and whether it was the
most prominent feature. The paper reports a curated 960-video test set with 60
videos in each of 16 action classes. These labels are relevant to audiovisual
event decisions, but this does not show that both modalities are required for
any one label. Any future task must compare audio-only, video-only and matched
AV, with source-video grouping and mismatch/time controls.

The published 960-video test set and reported results are already public and
observed; do not treat them as a blind audit or use them for tuning. The rating
table identifies videos using the original Moments in Time class-folder and
filename scheme. Underlying content deduplication and a project-specific
train/calibration/dev split were not audited here.

## Access and representation limitations

- The Zenodo record [10.5281/zenodo.8253350](https://zenodo.org/records/8253350)
  lists rating/test metadata and two embedding archives (approximately 0.50 GB
  and 2.1 GB). The record page's `License` field is empty; no separate
  redistribution/training license for those archives was verified.
- The official repository says raw videos come from the Moments in Time dataset
  and require a separate access request to that source. No such request or
  access grant exists in this workstream. Rights in the original media remain
  unresolved for this use.
- The supplied features use VGGish/VGG16 or YAMNet/EfficientNet-B0, not the
  project's pinned Audio/Video Path A encoders. They are not a substitute for
  evaluating the project's candidate world representations.
- The source's test annotations and paper results are exposed. There is no new
  blind group available from this source screen.

## Disposition

**REVIEW / not admitted for training or quality evaluation.** AVMIT is a
promising task-design reference for audio/video correspondence, but access to
raw media, rights for the deposited feature archives, project-specific unique
grouping and a fresh independent evaluation set are unresolved. Do not fetch
the archives, inspect the public test labels, or fit a readout. The STARSS23
result remains a negative, task-specific development result and is not replaced
by this source lead.

Primary references:

- [Zenodo record](https://zenodo.org/records/8253350)
- [Open-access AVMIT paper and data-availability statement](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301098)
- [Official repository and MIT source-media access note](https://github.com/mjoannou/audiovisual-moments-in-time/tree/e3c06ec0e215b1934d141f86008e92e567366d6c)
