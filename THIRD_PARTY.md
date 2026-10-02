# Third-party models and datasets

This repository is Apache-2.0, but upstream models and datasets retain their own licenses.

## Base model candidate

- Repository: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Intended role: pretrained multimodal alignment backbone
- Upstream attribution and NOTICE requirements must be copied into release artifacts as required by the pinned upstream revision.

No upstream model weights are stored in this Git repository.

## Datasets

Every dataset source must have a manifest recording:

- repository/dataset ID
- exact revision
- subset/split
- modality
- license
- whether redistribution of media is permitted
- whether commercial use is permitted
- whether it is training-only or evaluation-only

A permissive model license does not override dataset restrictions.

## Release rule

Do not publish a trained checkpoint until all contributing dataset sources have been reviewed and the release manifest contains their provenance and applicable notices.
