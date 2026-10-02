# Third-party models and datasets

This repository is Apache-2.0, but upstream models and datasets retain their own licenses.

## Base model candidate

- Repository: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Pinned revision: `6befbaca7398925921802abd1f277b495b78b738`
- Metadata license: Apache-2.0; the model card also points to [Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license).
- Attribution: Google DeepMind. Retain applicable license, attribution, and notice materials with distributed model-derived artifacts.
- Processor/tokenizer revision: same immutable revision as the model.
- Intended role: pretrained multimodal alignment backbone
- Upstream README at the pinned revision has no separate NOTICE file in the repository tree. Re-check upstream files and terms before redistribution.

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
