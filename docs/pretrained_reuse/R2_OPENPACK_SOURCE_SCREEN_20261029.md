# OpenPack synchronized action data source screen — 2026-10-29

## Decision

`REVIEW_NOT_ADMITTED` for this product-directed research track. OpenPack is
task-relevant for video/sensor action decisions, but the official dataset terms
exclude commercial use. The RGB release is even narrower: academic research
only, no commercial use, and no redistribution. The official benchmark split
also has participant overlap between train and validation. Do not fetch or
fit this data under the current product-oriented source policy.

No media, labels, or payload files were downloaded or opened. Only public
official documentation was read. No account, request, or license acceptance
was submitted.

## Pinned source evidence

- Official repository: `open-pack/openpack-dataset`, `main` resolved by
  `git ls-remote` to `0485a97c508c74018320dfc3eb90a1b6bc2d7d54`.
- The official repository README says OpenPack contains 53 hours of human
  movement data and 20,129 work-operation annotations. The official project
  page lists 16 participants, wearable IMU/E4 sensors, front/top depth cameras,
  LiDAR, and IoT data. It describes work-activity recognition as the central
  task, which is a plausible typed Decision target.
- The official dataset page says the dataset without RGB is CC BY-NC-SA 4.0.
  The separate official RGB terms limit use to academic research, prohibit
  commercial use, and prohibit redistribution, including when combined with
  models or programs. These terms do not meet the current product-directed
  source policy.
- The public OpenPack Challenge 2022 split assigns different sessions to
  train/validation/test, but participants `U0101`, `U0103`, `U0105`, `U0107`,
  and others occur in both train and validation. It is not a participant-
  disjoint generalization test. A new group split could be designed only if the
  source were otherwise admissible.
- The official page advertises a preprocessed four-IMU-with-labels archive of
  515 MB. The current C: free space was approximately 439 MB, also below that
  single-file size; no transfer was attempted.

## R2 disposition

OpenPack does not resolve the open R2-M requirement for a product-admissible,
group-disjoint, synchronized multi-input task. It is not evidence for or
against frozen multimodal fusion quality. Keep the candidate excluded unless
the project's source policy changes with explicit authorization or the
provider publishes compatible terms. No GPU, cloud, or storage resources were
used for this screen.

## Primary references

- [Official OpenPack dataset README at pinned commit](https://github.com/open-pack/openpack-dataset/blob/0485a97c508c74018320dfc3eb90a1b6bc2d7d54/README.md)
- [Official benchmark split at pinned commit](https://github.com/open-pack/openpack-dataset/blob/0485a97c508c74018320dfc3eb90a1b6bc2d7d54/docs/benchmark/data-split.md)
- [Official RGB terms at pinned commit](https://github.com/open-pack/openpack-dataset/blob/0485a97c508c74018320dfc3eb90a1b6bc2d7d54/licenses/OPENPACK_DATASET_RGB_LICENSE.md)
- [Official project overview](https://open-pack.github.io/)
