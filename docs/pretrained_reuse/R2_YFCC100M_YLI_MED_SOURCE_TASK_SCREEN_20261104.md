# YFCC100M / YLI-MED source and task screen — 2026-11-04

## Disposition

`REVIEW_NOT_ADMITTED` for R2-M. YFCC100M offers a large audio/video-adjacent
ecosystem and some CC0-derived features/annotations, but the official source
does not currently offer the core YFCC100M metadata through Webscope, original
media rights remain per-uploader, and the published YLI-MED labels do not
establish a task that requires both audio and video. Do not acquire or fit this
candidate for joint Decision quality on this evidence.

## Evidence checked

- The official Multimedia Commons core-dataset page says Webscope is not
  currently offering access to the YFCC100M metadata and has no reinstatement
  date. It reports roughly 15 GB for the compressed metadata file. The same
  page lists 787,479 videos in the public Multimedia Commons media bucket but
  says original media use is governed by the Creative Commons licenses chosen
  by individual uploaders and that license information is in the YFCC metadata.
- The AWS Open Data Registry says the derived audio/visual feature corpus and
  its annotations are CC0. That statement applies to those derived artifacts;
  it does not replace the per-uploader terms on original images/videos or
  establish that a specific media subset is eligible for this project.
- The official YLI-MED page describes about 2,000 positive videos across ten
  event classes and about 48,600 negatives, plus annotations such as spoken
  language and presence of musical score. Its listed audio features are pitch
  and MFCC; visual features include AlexNet and LIRE. The event labels are
  video-event labels, while language/music labels are audio-oriented. The
  source does not establish an outcome for which both modalities are necessary.
- The official YFCC challenge describes photographer/user-grouped partitions,
  which is useful precedent, but this does not establish a complete, current,
  independently grouped YLI-MED split or resolve the unavailable metadata
  access needed to check contributor/asset overlap.

No media, feature payload, annotation payload, metadata archive, model, or
evaluation set was downloaded or opened for this screen. No training or
inference was performed. The screen uses official web documentation inspected
on 2026-11-04; no immutable YFCC metadata revision was available to pin.

## Gate assessment

| Gate | Evidence | Status |
|---|---|---|
| Rights-clear original media for intended use | Per-uploader license per item; required metadata unavailable through stated Webscope route | **Unverified** |
| Rights-clear derived artifacts | AWS Registry identifies derived features/annotations as CC0 | **Potentially usable as artifacts, not proof of raw-media rights** |
| Independent grouping | YFCC challenge used photographer grouping; exact current YLI-MED split/group coverage not checked | **Unverified** |
| Frozen Decision task | Event labels plus audio attributes are listed | **Task family exists** |
| Both-input necessity | Event labels are video-centric; language/music labels are audio-centric; no both-required target/control evidence | **Not established** |
| Local feasibility | Metadata acquisition unavailable as documented; feature subset/schema/size not audited | **Not established** |

## Decision

Do not use YFCC100M/YLI-MED as the next R2-M training source. A future
reconsideration would require a currently accessible, rights-auditable metadata
route; an explicit allowed per-item license filter or a task restricted to
rights-clear derived features; pinned label and group manifests; and a
predeclared task with option-/metadata-only and matched unimodal controls that
can test whether both inputs are genuinely needed. The public event labels may
be screened separately as a video-only task only if that work is authorized
under the applicable source terms and a valid fresh split can be constructed.

## Primary sources

- [Multimedia Commons YFCC100M core dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)
- [AWS Open Data Registry: Multimedia Commons](https://registry.opendata.aws/multimedia-commons/)
- [YLI-MED corpus documentation](https://multimediacommons.wordpress.com/yli-multimedia-event-detection-subcorpus/)
- [YFCC100M user-grouped challenge description](https://multimediacommons.wordpress.com/tag-caption-prediction-challenge/)
- [Yahoo/Flickr YFCC100M overview](https://code.flickr.net/category/labs/)
