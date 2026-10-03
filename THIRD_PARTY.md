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

Verified dataset candidates and exact revisions are recorded in [docs/DATASETS.md](docs/DATASETS.md).
The approved initial text training source is `n4ze3m/typed-decisions-synth` at
`5ece89a225b23c4cd5c4bab5735a0819d61dd7d5`, metadata license MIT. Its card says the
data and labels are LLM-generated and unreviewed; its use is a controlled synthetic
training candidate, not a factual or quality benchmark. Open-Jev's redistributable
text release is also pinned there. OneJev is mixed-license and remains REVIEW until
source-level terms are represented and approved. MMAU test-mini is CC-BY-NC-4.0;
MVBench's MIT annotation license does not grant commercial redistribution or training
rights to its third-party videos. Those benchmarks remain evaluation-only.

Oxford Clevr-4 is an image training candidate under the official CC BY 4.0 terms, with
attribution to Sagar Vaze, Andrea Vedaldi, and Andrew Zisserman and citation of *No
Representation Rules Them All in Category Discovery* (NeurIPS 2023). The approved train
manifest pins the official image archive checksum; the held-out `val` split is evaluation-only.
This is a controlled synthetic image-classification source and does not clear rights for
unrelated natural-image datasets.

Google Speech Commands v0.02 is a keyword-audio training candidate under CC-BY-4.0. Its
Hub snapshot and per-shard Parquet SHA-256 values are pinned in the train and test
manifests. Cite Pete Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech
Recognition* (2018). Use only the ten keyword labels; do not attempt to identify speakers.

CLEVRER is a synthetic video candidate under the official CC0 grant. Cite Kexin Yi et al.,
*CLEVRER: Collision Events for Video Representation and Reasoning* (ICLR 2020). Its
question JSON hashes, official split and video archive references are pinned in the
manifests. Video bytes are not redistributed by this repository.

## Release rule

Do not publish a trained checkpoint until all contributing dataset sources have been reviewed and the release manifest contains their provenance and applicable notices.
