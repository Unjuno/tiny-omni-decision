# Open Images V6 train-frozen target scope (2026-10-26)

## Result

A train-only scan froze 11 hierarchy-matched object categories for a possible
four-choice image Decision task. Only after the class list was frozen did a
separate validation feasibility scan measure support for those same categories.
No images were downloaded, no model was loaded, and no quality metric or
checkpoint was evaluated.

The task contract is: one image, the fixed question `Which object is present in
the image?`, and four sibling class names. The target must have a
`Source=verification`, confidence-1 label; all three distractors must be
confidence-0 sibling labels. Unmentioned classes are never treated as absent.
At most one task is emitted per image when sampling a split.

## Frozen target selection

The target rule was fixed before reading validation labels:

- count unique training images with a positive target and at least three
  explicit negative siblings from the target's deepest qualifying hierarchy
  parent;
- require at least 1,000 such train images;
- remove display names containing any of the following complete-word tokens:
  `arm`, `baby`, `body`, `boy`, `child`, `ear`, `eye`, `face`, `finger`,
  `foot`, `girl`, `hair`, `hand`, `head`, `human`, `leg`, `man`, `mouth`,
  `neck`, `nose`, `person`, `skin`, `teeth`, `tooth`, `woman`;
- retain the full selected set regardless of its later validation support.

This is a fixed train-only scope rule, not a claim that the 11 labels are the
only useful Open Images classes. The source hierarchy repeats some label IDs
under multiple parents; the implementation selects each label's deepest
qualifying sibling group and resolves equal-depth ties lexicographically. This
handles the observed DAG structure without counting one class as multiple
targets. The resulting train support (54,436 images and 116,620 target/image
pairs before class/name filtering) matches the previous full-scan aggregate.

| Frozen target | Unique train support | Validation support |
|---|---:|---:|
| Toy | 1,666 | 98 |
| Door | 1,210 | 48 |
| Food | 1,406 | 440 |
| Plant | 3,801 | 621 |
| Sculpture | 1,138 | 90 |
| Vehicle | 2,353 | 85 |
| Clothing | 8,336 | 824 |
| Furniture | 2,676 | 179 |
| Building | 2,220 | 207 |
| Window | 2,311 | 105 |
| Animal | 5,708 | 655 |

Validation support totals 2,941 unique images across the fixed categories.
The support counts above are per-target counts; an image can support more than
one target before the later one-task-per-image sampler is applied. Validation
was used only for feasibility here. No class was removed, added, or reordered
based on these counts.

## Reproduction and hashes

The code is in [`openimages_options.py`](../../src/tiny_omni_decision/openimages_options.py),
[`audit_openimages_v6_train_support.py`](../../scripts/audit_openimages_v6_train_support.py),
and [`audit_openimages_v6_validation_support.py`](../../scripts/audit_openimages_v6_validation_support.py).
CPU tests cover explicit-negative handling, hierarchy DAG resolution, fixed
train-only class selection, deterministic image-unique sampling, and validation
support against an explicitly supplied frozen class list.

Train audit command:

```powershell
python scripts/audit_openimages_v6_train_support.py `
  --train-images <pinned-train-image-metadata.csv> `
  --train-labels <pinned-train-human-labels.csv> `
  --hierarchy <pinned-label-hierarchy.json> `
  --class-descriptions <pinned-class-descriptions.csv> `
  --output <train-support.json>
```

Validation feasibility command:

```powershell
python -m scripts.audit_openimages_v6_validation_support `
  --validation-images <pinned-validation-image-metadata.csv> `
  --validation-labels <pinned-validation-human-labels.csv> `
  --hierarchy <pinned-label-hierarchy.json> `
  --train-support-report <train-support.json> `
  --output <validation-support.json>
```

The local runs used conditional requests against the recorded GCS ETags and
generations. No raw row, image ID, original URL, author, or author profile was
included in the committed summaries.

| Input | GCS generation | SHA-256 |
|---|---:|---|
| Train image metadata | `1526567626302755` | `05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0` |
| Train human image labels | `1580742923196487` | `a97b079ccccf635a980bfe4b8180a0c844b4d9c887b2b7bc56535f539af044ea` |
| Label hierarchy | `1528191513246771` | `a28f06e83c8378c4285d92d768ceda7568a48cac3c02994b6c346df804e7d612` |
| Class descriptions | GCS ETag `63832b86bd4ce48c82825826c78ab15c` | `f7b528006a95db15fee1af30b198422ab067cee060f496e7a9f8930e5624d209` |
| Validation image metadata | `1526567618256414` | `ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17` |
| Validation human image labels | `1557303846177611` | `c2bc8b312ee7ed01ac4c58a62c2432c7f6b83b320874cbb0644e2f51aed3faad` |

The machine-readable train-only target/support table is
[`openimages-v6-train-target-support-20261026.json`](../../manifests/pretrained_reuse/openimages-v6-train-target-support-20261026.json)
(SHA-256 `a3a51bc54a1f14b3f584a7a9b44ba1a220cc7783b9480809ddfa3325f162cab3`).
The separate validation feasibility record is
[`openimages-v6-validation-target-support-20261026.json`](../../manifests/pretrained_reuse/openimages-v6-validation-target-support-20261026.json)
(SHA-256 `87e20327d5522d0372dbf5f5e2e59c6e5a8f36113b975746459dd6580fbe37f0`).
The exact validation class-list hash is
`56afa73d4db5e286da9baf1c813ac9fb62b21f12b77f310789831b4344503b88`.

## Limits and next gate

Open Images image-level labels are non-exhaustive. Requiring explicit negative
labels reduces false negatives but does not make the task complete or
unambiguous. The official metadata says images are listed as CC BY 2.0, but
Google disclaims individual ownership/license verification. **No image is
cleared for acquisition or training yet.** Before sampling or downloading
media, verify each selected landing page, attribution, source MD5, and
availability; preserve a private attribution receipt; then check train/validation
source-MD5, perceptual-hash, and normalized-content disjointness. Only after
those gates may the pinned frozen-image Path A be run. The 11 classes and
validation support are a feasibility result, not model accuracy evidence and
not evidence for the product's 95–98% target.
