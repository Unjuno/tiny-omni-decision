# R2 Localized Narratives / COCO source screen — 2026-11-13

## Decision

**Status: `REVIEW_NOT_ADMITTED`.** Localized Narratives has a plausible
audio+image grounding task, but this metadata-only screen does not establish
usable image rights/attribution, voice-recording rights and consent scope, or
an independent speaker split. Do not fetch image/audio payloads, construct a
Decision corpus, fit a readout, or score a model from this candidate until those
gates are resolved. This is a source screen, not quality evidence.

## Candidate task and fit

The official source describes a spoken image narrative synchronized with mouse
traces. A possible R2-M task is to choose which of four image regions is being
described by a fixed spoken segment at a known time. Such a task can require
both the image and audio and supports audio-only, image-only, and mismatched
input controls. It is an image+audio task, not video understanding, and has not
been admitted or evaluated.

## Sources and pinned local metadata

- Google Localized Narratives page: <https://google.github.io/localized-narratives/>.
  It reports 134,272 COCO train and 8,573 COCO validation annotations; records
  include `image_id`, `annotator_id`, `caption`, timed utterances, traces, and a
  relative voice-recording path. It releases annotations under CC BY 4.0 and
  requests attribution to the paper.
- COCO terms: <https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm>.
  They state that COCO does not own image copyright and that image use must
  follow Flickr terms.
- Metadata-only local screen directory:
  `D:\CodexArtifacts\pretrained-reuse-r2\localized-narratives-coco-metadata-source-screen-20261113-attempt-01`.
- Localized Narratives train shard 0:
  `coco_train_localized_narratives-00000-of-00004.jsonl`, 33,573 rows,
  1,362,414,015 bytes, SHA-256
  `EAF29A1A9CACD3A84E7D50B8041E9F6363364CE6BD338E34637961A13BE70FC7`.
- Localized Narratives COCO validation JSONL:
  `coco_val_localized_narratives.jsonl`, 8,573 rows, 351,718,628 bytes,
  SHA-256 `4EE454CAEC23442D0D1A43D0B35A714D74FC5E6CCC4E6F240D6E86B2FB709542`.
- Official COCO train/val annotations archive:
  `annotations_trainval2017.zip`, 252,907,541 bytes, SHA-256
  `113A836D90195EE1F884E704DA6304DFAAECFF1F023F49B6CA93C4AAAE470268`.
  It was inspected in place; no members were extracted to the repository.
- Summary SHA-256: `43ABE56910CF54243AEB49C8CF234998F6758BEB66941D754A98628DA7A9A282`.

No audio, image, model, prediction, test, or sealed-audit payload was loaded.
The local artifact directory is outside Git and is not copied into the
repository.

## Metadata findings

| Check | Train shard 0 | Official validation |
|---|---:|---:|
| Rows | 33,573 | 8,573 |
| Unique image IDs | 31,756 | 5,000 |
| Unique annotators | 108 | 100 |
| Voice-recording paths present | 33,573 | 8,573 |
| Unique normalized captions | 33,552 | 8,572 |
| COCO license ID 4 (CC BY 2.0) | 5,558 | 1,465 |
| License ID 7 (no known copyright restrictions) | 159 | 5 |

The two official splits have zero image-ID overlap, but all 100 official
validation annotator IDs also occur in train shard 0. Two normalized captions
overlap. Consequently the official validation split is not an
independent-speaker validation set and must not be used as one. Whether a new
speaker-disjoint and image-disjoint split can be formed from untouched train
records has not been established; historical project exposure is also not
fully reconciled for this source.

The COCO captions metadata includes `flickr_url`, `coco_url`, and a license ID,
but no author/owner field. All 118,287 `flickr_url` values in the inspected
captions metadata point to `staticflickr.com` image paths; none is a Flickr
author/photo page. Therefore the available metadata does not support
per-image creator attribution for a candidate subset. A COCO license ID alone
does not resolve each image owner's rights or attribution obligations. The
annotation page's general CC BY 4.0 statement also did not independently
resolve the rights/consent scope for crowd-sourced voice recordings.

## Gate outcome and next step

The task shape is promising for testing whether synchronized audio adds
information to visual localization, but rights and independent-group gates
fail closed today. Preserve the downloaded metadata-only screen as external
provenance. Reconsider only if authoritative per-image attribution/rights and
voice-recording rights/consent are established, then reconcile source exposure
and prove a deterministic annotator- and image-disjoint train/development split
before downloading any media. Do not spend GPU time on this candidate now.
