# X-Capture bounded media/data-quality audit — 2026-11-28

## Scope and frozen selection

This is a train/validation-only input-quality screen for the narrow
cross-sensory object-identity retrieval task in
[`R2_XCAPTURE_SOURCE_SCREEN_20261128.md`](R2_XCAPTURE_SOURCE_SCREEN_20261128.md).
It is not a model-quality experiment. No checkpoint, feature encoder, GPU, or
evaluation partition was used.

The audit used the pinned `swistreich/XCapture` archive revision
`b4468bf999aac250f34ef1b02459d902492a068f`, archive SHA-256
`71855c2d9cdfb6d54ed379ff4059b6c370e6f7ef040830b0ac19063159c477c2`, and
central-directory SHA-256
`3d8e546f1ca55e52a2b8e6c72aab99d14815e9da631589cb4d144f1179790f17`.
The existing seed-17 split assigns complete object groups, not points, to
400 train / 100 validation / 100 evaluation IDs. A deterministic hash-ranked
sample selected one point from each of 10 distinct train objects and 10
distinct validation objects. No evaluation ID or payload was accessed.

| Partition | Sampled object/point IDs |
|---|---|
| Train | `00134/005`, `00083/001`, `00315/004`, `00159/005`, `00155/000`, `00523/001`, `00352/001`, `00471/003`, `00061/005`, `00218/003` |
| Validation | `00104/003`, `00394/001`, `00502/005`, `00048/002`, `00118/001`, `00488/002`, `00429/001`, `00154/005`, `00424/004`, `00095/002` |

For each point, RGB, normalized audio, 10N tactile image, and
`point_cloud/mask.png` were fetched by HTTP range from the pinned archive. The
80 selected files passed local ZIP CRC32 and declared-size validation. The
sample receipt SHA-256 is
`17bacf80527a1aa0e1496eb330637d7f81b039ca68ffa28ccfabf9b33b240c0f`.

## Findings

The paper says the point-cloud mask is produced from a SAM mask queried around
the centered target point, followed by erosion and application to the depth
map. The local preprocessing routine enforces dimensions and binary values,
zeros image pixels outside the mask, and crops the mask box with 8% padding.
Across 20 sampled points, mask coverage ranged from 4.60% to 59.78%, with a
mean of 18.98% and median of 16.59%.

Manual contact-sheet inspection found:

- At least 5 of 20 masked crops still show readable package text, a barcode,
  or a brand mark. Some markings are on the target package/object itself.
- A few crops retain disconnected mask fragments or pixels from nearby scene
  surfaces. The mask reduces background content but does not remove all of it.
- Several targets are composite or labeled products. Their writing is part of
  the observed object and cannot simply be called leakage.

The count is a small manual screen, not an OCR result or a corpus-wide rate.
Because split groups are whole object IDs, the same exact object identity and
its six points do not cross train, validation, and evaluation. Nevertheless,
text and surrounding-scene cues need a frozen ablation before any model result
can be attributed to cross-sensory alignment.

## Reproducible artifacts and limits

- Media sample: `C:\CodexArtifacts\pretrained-reuse-r2-local\xcapture-sample-audit-20261128-attempt-04`
- Masked-crop report: `C:\CodexArtifacts\pretrained-reuse-r2-local\xcapture-masked-crops-20261128-attempt-02\masked-crops.json`
- Masked-crop report SHA-256: `6c37406b3e8011bab5fd564470b5b8d312d4016966906a1f2683f723d4453c84`
- Contact sheet: `C:\CodexArtifacts\pretrained-reuse-r2-local\xcapture-masked-crops-20261128-attempt-02\contact-sheet.png`
- Preprocessing: `src/tiny_omni_decision/xcapture_preprocess.py`
- Focused preprocessing tests: 5 passed.

This screen does not test synchronization quality beyond the four-channel
point association in the archive paths, does not establish the semantic
contents of the author's object descriptions, and does not measure retrieval
quality. Before fitting, freeze a retrieval candidate gallery and compare
audio-only, image-only, tactile-only, and joint arms with the same object and
point sampling; add zeroed/shuffled modalities and a text/mark-occlusion
diagnostic. Keep validation exposed to development and reserve the 100-object
evaluation IDs for one frozen final evaluation only.

The official source and paper define the dataset and target protocol:
[pinned X-Capture dataset card](https://huggingface.co/datasets/swistreich/XCapture/blob/b4468bf999aac250f34ef1b02459d902492a068f/README.md),
[ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Clarke_X-Capture_An_Open-Source_Portable_Device_for_Multi-Sensory_Learning_ICCV_2025_paper.pdf).
