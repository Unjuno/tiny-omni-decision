# R2 Open Images V6 fixed pilot task manifests (2026-10-26)

This records a deterministic, metadata-only pilot for the already frozen
11-class Open Images V6 scope. It is a task/data preflight, not a model result,
not a final evaluation set, and not authorization to acquire or train on the
images.

## Construction

The train-only target scope in
[`R2_OPENIMAGES_V6_TARGET_SCOPE_20261026.md`](R2_OPENIMAGES_V6_TARGET_SCOPE_20261026.md)
was held fixed. The sampler used seed 17, at most 20 images per target, one task
per image within a split, explicit human verification labels, and hierarchy
sibling distractors. No target was dropped to improve the pilot. The selected
four-choice task manifests and private attribution receipts are stored outside
the repository under
`C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-support-audit-20261026\`.
They contain source identifiers and attribution data and must stay private.

| Split | Tasks | Per frozen target | Task manifest SHA-256 |
|---|---:|---:|---|
| Train | 220 | 20 × 11 | `d0710e9afad8368e76f0b86523e6241affd5c0f135d53cc55b3b0572cbe6342b` |
| Validation pilot | 220 | 20 × 11 | `8010407467c45fc944f9c6bf9c2303d2df2715a2786b24df44f7a50377e52309` |

ImageID overlap and original-file MD5 overlap between these selected splits
were both zero. The manifest sampler did not download media or load a model.
The source-MD5 check is an exact-file check; perceptual and normalized-content
deduplication has not yet run.

## Rights and media gate

The source metadata lists CC BY 2.0, but does not establish rights for each
asset. All 440 private receipt rows therefore remain
`individual_rights_verified=false` and `landing_page_check_status=not_checked`.
Fourteen selected rows lacked a 300K thumbnail URL; those records were retained
and the receipt points to the metadata's original-media URL instead. No image
was fetched. The subsequent per-page declaration check is recorded in the
[landing-page audit](R2_OPENIMAGES_V6_LANDING_PAGE_AUDIT_20261026.md): 347 pages
matched the CC BY 2.0 declaration and attribution profile, while 93 were
inaccessible or mismatched. The page check does not independently authenticate
copyright ownership, so no row is marked `individual_rights_verified=true`.
Perceptual-hash and normalized-content checks remain pending. A missing
thumbnail is not itself a reason to drop an otherwise selected example.

## Reproduction and implementation

The code is in [`sample_openimages_v6_tasks.py`](../../scripts/sample_openimages_v6_tasks.py)
and [`openimages_options.py`](../../src/tiny_omni_decision/openimages_options.py).
It refuses to overwrite outputs and refuses to write private attribution
receipts inside the repository. It accepts an absent thumbnail URL only when
the original URL and required attribution fields remain present, and records
which URL would be used. The sample IDs and private receipt rows are not
committed.

CPU tests cover deterministic per-split selection, fixed target scope,
image-unique tasks, required attribution fields, missing-thumbnail fallback,
and duplicate source-MD5 rejection. No GPU work, feature extraction, training,
checkpoint selection, or quality evaluation occurred in this step.
