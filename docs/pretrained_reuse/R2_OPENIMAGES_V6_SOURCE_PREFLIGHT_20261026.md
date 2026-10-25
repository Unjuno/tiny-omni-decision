# Open Images V6 natural-image source preflight (2026-10-26)

## Disposition

**Strong candidate for a bounded natural-image Decision benchmark, pending
per-image source verification and a train-split coverage audit.** This was a
read-only metadata/annotation preflight. No image bytes were requested, no
example rows or author names/profile URLs were retained, and no model was
run.

The official Open Images V6 download page links to image metadata that includes
image ID, split, original and landing URLs, license URL, author/profile,
original size/hash, and rotation. Its license page says annotations are CC BY
4.0 and images are listed as CC BY 2.0, while explicitly warning that Google
does not warrant individual image license status and users must verify each
image. Treat CC BY 2.0 as a filter, not a blanket guarantee. Any later
acquisition must preserve per-image landing URL, exact license, attribution,
and source checks; exclude missing, conflicting, inaccessible, or revoked
items. Do not store author/profile metadata in public artifacts beyond what is
needed for attribution receipts.

## Validation metadata snapshot

Official object:
`https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv`

The complete CSV was streamed into a temporary directory for aggregate checks;
the temporary copy was removed automatically. No image files were requested.

| Check | Result |
|---|---:|
| GCS object generation | `1526567618256414` |
| ETag | `643a54e43b0bab8acce8817b4a569780` |
| Download size | 15,245,485 bytes |
| SHA-256 | `ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17` |
| Rows / unique image IDs | 41,620 / 41,620 |
| Split values | validation: 41,620 |
| Image license field | 41,620 CC BY 2.0 |
| Original landing URL present | 41,620/41,620 |
| Author and author-profile fields present | 41,620/41,620 |
| Original URL and source MD5 present | 41,620/41,620 |
| Distinct author-profile values | 17,527 (counted transiently) |
| Duplicate source MD5 rows within validation | 0 |

This validates metadata schema/coverage for the official validation cohort; it
does not independently confirm ownership or continued availability at each
Flickr landing page. The training metadata object is 638,407,721 bytes
(generation `1526567626302755`; ETag `a4ac0bcedb5c2df4d1b1230dc6b41b8f`). It was
not downloaded or inventoried in this stage.

## Validation annotation feasibility

The V6 download page links validation human-verified image labels at:
`https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv`

That complete annotation file was likewise streamed to a temporary directory,
aggregated, and removed; no test annotations were accessed. Object generation
was `1557303846177611`, ETag `1897fce94618869aace7f60497517bdf`, size
24,824,095 bytes, SHA-256
`c2bc8b312ee7ed01ac4c58a62c2432c7f6b83b320874cbb0644e2f51aed3faad`.
It contains 595,339 label rows over 41,620 validation image IDs. The source
values were `verification` (592,817 rows) and
`crowdsource-verification` (2,522 rows). Considering only these explicit
human-verification sources, 33,561 images have at least one positive and at
least three explicit negative image labels; 661 have exactly one positive
label. This suggests a four-choice image-presence task can be constructed
without treating unannotated classes as negatives, subject to class-specific
support and source-image verification. Open Images image-level annotations are
not exhaustive; candidate negatives must be explicitly marked absent.

The 33,561 count is a feasibility aggregate, not an evaluated benchmark and
not a claim of accuracy. Validation was inspected only for source/task
feasibility; do not choose a trained checkpoint or tune hyperparameters using
this screen. Later define the class set from training-only counts, freeze a
separate validation subset, and leave official test untouched until a
predeclared final audit.

## Remaining gates and next step

1. From training metadata and labels only, estimate class support and build a
   candidate class shortlist. The official training files are large, so first
   measure transfer size/storage and stream only what is necessary.
2. Join retained train examples to exact license/attribution rows; verify each
   selected original landing page and image hash before retaining media.
3. Deduplicate by image ID and source MD5, then perceptual hash and normalized
   label/task content; keep image identities disjoint across train and
   validation. Do not access official test while selecting the task or model.
4. Only after those checks, acquire a small bounded CC BY-verified sample and
   evaluate the already-pinned VideoPrism image path with fixed supplied-option
   tasks. Record Accuracy, NLL, Brier, ECE, source coverage, and cost.

Do not promote this data source as product-cleared merely because its rows say
CC BY 2.0: the upstream page assigns responsibility for individual source
verification to the user.

## Official evidence

- [Open Images V6 download page and metadata schema](https://storage.googleapis.com/openimages/web/download_v6.html)
- [Open Images V6 dataset facts and license caveat](https://storage.googleapis.com/openimages/web/factsfigures_v6.html)
- [Open Images evaluation notes on non-exhaustive image-level labels](https://storage.googleapis.com/openimages/web/evaluation.html)
