# Existing train images: in-scope verified-label inventory

The [fixed protocol](R2_OPENIMAGES_EXISTING_TRAIN_RELABEL_PROTOCOL_20261027.md)
scanned the complete pinned human train-label object and verified its SHA-256.
All 176 pre-existing page-declaration-matched train images were found, and each
original target was a verified positive. Twenty-nine have at least three
explicitly verified negatives in the frozen 11-class scope; all 29 have a
corresponding existing VideoPrism train-feature row.

| Original target | Eligible images with existing features |
|---|---:|
| Toy | 1 |
| Door | 4 |
| Food | 1 |
| Plant | 3 |
| Sculpture | 1 |
| Vehicle | 1 |
| Clothing | 3 |
| Furniture | 5 |
| Building | 1 |
| Window | 9 |
| Animal | 0 |

These are availability counts, not a new task or a trained model. With the
original target assignment the existing 176-image train set cannot provide all
11 positive classes while requiring three in-scope negatives. Some images have
multiple positives; a different assignment would require a separately frozen
rule and verification, and this inventory alone does not establish that it
would solve role balance. The very small set is not a basis for model-quality
claims. Do not filter an existing validation set to match it.

The 176 page checks do not authenticate media ownership, and feature row
presence is not a new verification of the media bytes or split integrity.
Additional independently admitted training images and a prespecified balanced
construction are required before a representative experiment. No images,
validation examples or model weights were accessed by this audit.

The private result is
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-existing-train-inventory-20261027-attempt-01/result.json`,
SHA-256 `831ea63a80fc2d027ca17b514edcac399ccd3256305dbca0f55299f906308104`.
It pins the input, feature-row, scope, script and protocol hashes. Reproduce via
`python -m scripts.audit_openimages_existing_train_labels --output <new-json>`
with `PYTHONPATH=src;.` from the worktree.
