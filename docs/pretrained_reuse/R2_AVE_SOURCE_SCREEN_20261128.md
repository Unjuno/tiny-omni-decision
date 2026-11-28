# AVE audio-visual event source screen — 2026-11-28

## Disposition

**`REVIEW_NOT_ADMITTED` for product-directed training or quality selection.**
AVE is a strong task-design lead for synchronized sound-and-video event
decisions, but the pinned public materials inspected here do not state a
license for the dataset or its underlying clips, and they do not expose an
auditable official split manifest. No data archive, annotation payload, or
media was fetched or opened.

## Task and data evidence

The authors' project page describes 4,143 ten-second videos across 28
audio-visual event categories, with temporal event boundaries. The paper
defines an AV event as both visible and audible, and reports evaluating audio,
video, and joint inputs. This makes AVE relevant as a narrow audio-video event
Decision task, but does not establish that both modalities are necessary for
every category or example; any future fit would need matched unimodal and
input-permutation controls.

The authors' paper states that the videos originate from YouTube and that AVE
is a subset of AudioSet. The official project page and the pinned code
repository's README inspected for this screen contain no dataset-use license
grant. The pinned repository root contains no `LICENSE` file. A code license,
if added later, would not by itself resolve rights in the video/audio assets.

The official project page and pinned README inspected here do not provide a
hashable train/validation/test ID manifest or source-group rules. Split sizes,
cross-split clip identity, uploader overlap, and normalized audio/video overlap
therefore remain **UNKNOWN** in this screen. Do not infer them from third-party
mirrors or copied split lists.

## Decision and next gate

Do not download, reconstruct, or train on AVE under the current source gate.
Reconsider only after the rights holder or publisher supplies explicit
product-development training/evaluation permission for the underlying media,
and an official split/identity manifest that can be audited by video and
source group. A permitted split must preserve every source video in exactly
one fold, then pass media/content overlap checks before any feature extraction.

This screen does not reject the event-localization task itself; it rejects
AVE as an admitted data source on the evidence currently available. No model,
checkpoint, data manifest, or evaluation result was changed. No GPU or cloud
compute was used.

## Pinned evidence

- Authors' project page: <https://sites.google.com/view/audiovisualresearch>
- Authors' paper: <https://arxiv.org/abs/1803.08842>
- Authors' code repository at commit
  [`93f843fed72fb17394df76330df958b0c2dbde76`](https://github.com/YapengTian/AVE-ECCV18/tree/93f843fed72fb17394df76330df958b0c2dbde76)
- Pinned README SHA-256:
  `5c848c1026583d14727e95a6f116f1c3dac4ee0f7129397a7c1f7e1a100ac777`
- Pinned README bytes: `4486`
- Observed: 2026-11-28; metadata/repository inspection only.

Machine-readable record: [`ave-source-screen-20261128.json`](../../manifests/pretrained_reuse/ave-source-screen-20261128.json).
