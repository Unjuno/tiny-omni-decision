# LINE Japanese DistilBERT Runtime Audit — 2026-11-23

## Candidate and source

Candidate: `line-corporation/line-distilbert-base-japanese`, pinned Hub
revision `c32d05753d0d4be0f60482291a9452ec7e6f7d68`. The Hub declares
Apache-2.0. Its model card describes a six-layer, 768-hidden, 12-head
DistilBERT pretrained on 131 GB of Japanese web text, and reports JGLUE
results including 88.9 JNLI accuracy and 76.1 JCommonsenseQA accuracy. Those
are upstream benchmark results, not this project's Decision-task results.

The frozen checkpoint contains **68,712,704 parameters** and its FP32
`pytorch_model.bin` is 274,894,944 bytes (SHA-256
`9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`). The
configuration has vocabulary 32,768 and a 512-token position limit. This is a
smaller Japanese BERT-family representation than the 110.6M Tohoku BERT-base
already compared on the typed-decision probe. It leaves more of the provisional
parameter budget for non-text towers, but storage and total multimodal size
still require measurement.

## Local execution

Weights were downloaded to the new external directory
`C:\CodexArtifacts\pretrained-reuse-r2-local\line-distilbert-ja-20261123-attempt-01`
at the pinned revision. Every downloaded file was hashed; the weight SHA-256
matches the Hub LFS object digest. No model artifact was committed to Git.
The external machine-readable smoke record is
`r1-line-distilbert-runtime-audit.json` (SHA-256
`ff2654d2fd89a2a374d74d7079db1fb5703791114199c441924bb44986899f66`). The
source-of-truth candidate inventory now includes this checkpoint; its updated
SHA-256 is `a84288f12861d3bbcf0c5ecd823de4872c0750862b3eb42156829b119659045b`.

For the CPU load smoke, PyTorch's `weights_only=True` loader read the checkpoint
as a tensor-only state dictionary. The standard Transformers
`DistilBertForMaskedLM` architecture loaded it with `strict=True`; the model was
then frozen and its `distilbert` encoder produced finite `[1, 4, 768]` output
in 0.0375 seconds with one CPU thread. This used no GPU and did not alter the
separate running Orbitune process. Exact loaded parameter count was
68,712,704; trainable count was zero.

The Hub tokenizer uses reviewed Apache-2.0 custom Python code, MeCab, and
SentencePiece with `unidic_lite`. These tokenizer dependencies are not installed
in the current environment, so **tokenization and end-to-end text inference
have not yet passed**. No remote code was executed. The checkpoint load smoke
does not establish Decision quality, calibration, or package fit.

## Decision and next gate

**Checkpoint feasibility: PASS. Encoder forward: PASS. Tokenizer: NOT RUN.
Decision quality: NOT MEASURED. Product/release eligibility: NOT ESTABLISHED.**

If proceeding, first create an isolated tokenizer environment pinned to this
revision and its declared dependencies, compare token IDs against the official
tokenizer on fixed Japanese fixtures, and run a small CPU-only smoke through the
same typed-decision input pipeline. Only after that passes should a paired
readout comparison be considered. The existing typed-decision validation is
already exposed and uses synthetic teacher distributions, so any reuse must be
labelled a diagnostic rather than independent confirmation; a new human-labeled
decision task remains necessary for a quality claim.

## Sources

- [Pinned model revision](https://huggingface.co/line-corporation/line-distilbert-base-japanese/tree/c32d05753d0d4be0f60482291a9452ec7e6f7d68)
- [Model card and declared JGLUE metrics](https://huggingface.co/line-corporation/line-distilbert-base-japanese)
- [Official LINE model repository](https://github.com/line/LINE-DistilBERT-Japanese)

