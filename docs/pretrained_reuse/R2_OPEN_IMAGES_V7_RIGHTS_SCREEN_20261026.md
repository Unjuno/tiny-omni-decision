# Open Images V7 natural-image source screen (2026-10-26)

## Disposition

**REVIEW; not admitted to training or evaluation.** This is a metadata-only
screen. No images, annotations, or evaluation examples were downloaded or
opened, and no model was trained.

Open Images V7 is a potentially useful natural-image source for typed visual
decisions: the publisher describes object boxes, visual relationships, masks,
point labels, and image-level labels over a large image collection. In
principle, object-relationship annotations could support answer-option
decisions without making prose captions the hidden input. This is only a task
candidate; label correctness, group-safe split construction, overlap control,
and model performance remain untested.

## Rights and data-integrity gate

The V7 publisher says annotations are CC BY 4.0 and images are listed as CC BY
2.0, but explicitly makes no warranty about the license status of each image
and asks users to verify every image's license. The V7 metadata format includes
an image ID, subset, original URL, landing-page URL, license URL, author,
title, original size, and original MD5. The metadata therefore enables
per-image investigation, but its presence alone does not satisfy the
publisher's verification instruction or the repository's per-source
admission gate.

Before reconsidering admission, a bounded candidate set would need to:

1. Verify the license on each original landing page and retain timestamped
   evidence, attribution, and image ID; reject inaccessible, ambiguous, or
   incompatible records.
2. Verify downloaded bytes against source metadata where possible, then hash
   the exact retained asset. Do not redistribute image bytes in the repo.
3. Build fresh train/validation groups by uploader/source identity, then check
   duplicate and near-duplicate image content across groups. The published
   split alone does not demonstrate those no-overlap properties.
4. Audit relationship/object annotations and predeclare a decision task,
   including distractor construction and image-only / matched text-control
   checks, before reading validation results.

Because per-image rights and no-overlap have not been demonstrated, this
source is not added to `manifests/training-candidates.yaml`. No asset is
approved by this screen.

## Official evidence

- [V7 description, dataset sizes, and license caveat](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [V7 downloads, annotations, metadata fields, and relationship data](https://storage.googleapis.com/openimages/web/download_v7.html)

Checked 2026-10-26. This screen does not alter the existing CLEVR-4 image
baseline or close R2 M-0/M-1.
