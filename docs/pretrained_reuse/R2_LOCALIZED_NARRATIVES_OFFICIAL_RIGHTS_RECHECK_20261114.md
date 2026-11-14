# Localized Narratives official rights recheck — 2026-11-14

## Decision

**Status remains `REVIEW_NOT_ADMITTED`.** The primary-source recheck did not
resolve either required media-rights gate. No media, model, prediction, or
evaluation payload was accessed; no training or GPU work was run. Do not use
the candidate for the proposed image+audio Decision task on the evidence
currently available.

## What the official sources establish

- The [Google Localized Narratives page](https://google.github.io/localized-narratives/)
  says that its **annotations** are released under CC BY 4.0 and asks users to
  attribute the paper. It lists `voice_recording` as a path to an OGG file, but
  the page does not explicitly state that the voice recordings themselves are
  covered by the annotation license or provide a separate recording-rights or
  speaker-consent grant. Public availability of the OGG path is not evidence of
  that missing grant.
- The [COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
  say COCO does not own image copyrights and direct users to follow Flickr
  terms. The existing metadata screen found `flickr_url` values that point to
  static image paths, not Flickr photo/creator pages, and no author field.
- Flickr's current [image-use guidance](https://www.flickrhelp.com/hc/en-us/articles/10710266545556-Using-Flickr-images-shared-by-other-members)
  says uploads default to All Rights Reserved and directs users to the photo
  page for copyright/license information. The [Flickr license guide](https://www.flickrhelp.com/hc/en-us/articles/40048459562388-Copyright-Licenses-on-Flickr)
  likewise identifies each photo page as the place where its license appears.
  The currently pinned COCO metadata therefore does not provide the per-image
  license and attribution evidence needed for a selected training/evaluation
  subset.

These observations establish **insufficient evidence for admission**, not a
legal conclusion that every individual image or recording is unusable.

## Split and exposure gates remain closed

The prior metadata-only screen found zero image-ID overlap between the official
COCO train and validation annotation files, but all 100 official-validation
annotators also occur in train shard 0. It did not establish a fresh
annotator-disjoint and image-disjoint split across all train shards or reconcile
historical project exposure. The roadmap's provisional independent-group
minimum is 200; no qualifying split has been demonstrated.

The proposed task—selecting the image region described by a synchronized spoken
segment—remains a plausible both-input-required task, but task plausibility
does not cure rights or split gaps.

## Reopen criteria

Reopen only after authoritative source/provider documentation establishes:

1. recording-level rights and speaker-consent scope sufficient for the intended
   model-development and product use; and
2. a per-image rights/attribution route that can be mapped to every selected
   COCO image, or a different image source with equivalent rights evidence.

Only then perform a metadata-only full-shard and exposure audit. Require a
deterministic speaker/annotator- and image-disjoint train/development split,
zero relevant identity overlap, and the predeclared independent-group floor
before accessing media or fitting/scoring a model. If either rights requirement
cannot be established, close this candidate and move to another source; do not
infer permission from public URLs or the annotation CC BY label.

## Sources checked

- Google Localized Narratives: https://google.github.io/localized-narratives/
- COCO terms: https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm
- Flickr image-use guidance: https://www.flickrhelp.com/hc/en-us/articles/10710266545556-Using-Flickr-images-shared-by-other-members
- Flickr license guide: https://www.flickrhelp.com/hc/en-us/articles/40048459562388-Copyright-Licenses-on-Flickr
- Prior pinned metadata and split counts: `R2_LOCALIZED_NARRATIVES_SOURCE_SCREEN_20261113.md`
