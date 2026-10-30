# AVA Active Speaker source screen — 2026-10-18

## Decision

Keep AVA Active Speaker as a **review-only candidate** for a bounded audiovisual
speaker-to-face association task. Do not download or use its media yet. The official
Google Research page says the label dataset is CC BY 4.0, but the referenced media are
original YouTube videos and their per-source rights were not resolved by this screen.
No local annotation archive, media, split manifest or source hash was inspected; the
candidate manifest records these fields as unknown.

## Why the task is relevant

The official v1.0 labels identify, for each face in each frame, one of
`SPEAKING_AND_AUDIBLE`, `SPEAKING_BUT_NOT_AUDIBLE`, or `NOT_SPEAKING`. Different
people have separate rows, with an `entity_id` linking a person's boxes across nearby
frames within a video. The official release says it covers 160 videos still available
from an original list of 188; its train/validation partitions follow AVA Actions.
Google reports roughly 3.65 million labeled frames over 38.5 hours, and describes a
joint audiovisual model with gains from audiovisual modeling and temporal integration.
These are source/task descriptions, not performance estimates for our candidate.

A useful Decision view would take synchronized audio/video plus an externally
provided face-track proposal and return calibrated probabilities for whether that
candidate is speaking and audible. Audio without the candidate's visual location
cannot associate a voice with a face; video alone cannot establish audibility. This
makes the task more relevant to audiovisual correspondence than AV quality ratings.
However, the dataset's labels do not prove that every example requires both streams:
visual mouth motion can reveal speaking, while audio can reveal speech presence.
Only stratified modality ablations and carefully defined candidate association can
measure their complementary contribution. Supplied face boxes also make this an
oracle-proposal component test, not end-to-end person detection or the general Omni
Decision product.

## Rights, grouping and integrity gates

- The official page describes the AVA dataset release as CC BY 4.0 and links separate
  train and validation label archives. This screen did not download them, so archive
  revisions, hashes, file rows and video-ID disjointness remain unknown.
- The media are YouTube movie/video assets. The source page's dataset-level CC BY
  statement does not settle the rights or continued availability of each underlying
  recording for our use. Treat media as `PER-SOURCE` / `REVIEW`; no fetch, feature
  extraction or training is authorized by this screen.
- Before any future use, verify official train/validation video IDs, exact and
  normalized content overlap, within-video entity grouping, and any available
  participant/source grouping. Keep all rows from a video in one split. Keep challenge
  and test labels sealed from iteration.
- Use no face-identity embeddings or identity recognition. Count any external face
  detector/proposal module in the total resident parameter and runtime budget.

## Possible fixed evaluation, only after rights clearance

Predefine the same candidate face/time rows for query-only, audio-only, video-only,
and audiovisual controls. Preserve the three upstream label meanings; do not merge
`SPEAKING_BUT_NOT_AUDIBLE` with `NOT_SPEAKING` without a separately specified binary
task. Report class-balanced accuracy, per-video macro metrics, NLL, Brier, ECE, and
grouped uncertainty. Missing, time-shifted, and cross-recording audio controls must
not reuse old labels if the change can alter truth; such examples need valid relabels
or must be described as sensitivity diagnostics only.

This screen does not establish that the existing conditional AV gate is useful or
needed. It only identifies a potentially better-labeled source for a later, rights-
cleared, bounded AV correspondence test. All model and product quality gates remain
open.

## Sources

- [Official AVA download and label schema](https://research.google.com/ava/download/)
- [Google Research AVA-ActiveSpeaker paper summary](https://research.google/pubs/ava-activespeaker-an-audio-visual-dataset-for-active-speaker-detection/)
- [AVA-ActiveSpeaker paper](https://arxiv.org/abs/1901.01342)
- [CVDFoundation download instructions at pinned commit `70340626184cbde175cb1359bdf5af6f75e28231`](https://github.com/cvdfoundation/ava-dataset/tree/70340626184cbde175cb1359bdf5af6f75e28231)

The pinned CVDFoundation commit identifies only the public download instructions;
it is not a version/hash for the label archives or original media. No archive/media
hashes or training artifacts are claimed by this screen.
