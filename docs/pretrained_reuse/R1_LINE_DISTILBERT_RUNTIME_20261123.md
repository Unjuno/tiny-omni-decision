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
`d62486b7ec2ad59c0d7d9a644b0a44776c536de5e393b49e0834fc89fcd79c57`). Its
tokenizer subrecord is `tokenizer-forward-smoke.json` (SHA-256
`d8be9421f60eda085eccd528a7803940feac038e4567a56922456bdc12158364`). The
source-of-truth candidate inventory includes this checkpoint; its updated
SHA-256 is `640418d297af45fa97d3f2577c17a652dc7e63cad2fcf0e904525841e54fad22`.

For the CPU load smoke, PyTorch's `weights_only=True` loader read the checkpoint
as a tensor-only state dictionary. The standard Transformers
`DistilBertForMaskedLM` architecture loaded it with `strict=True`; the model was
then frozen and its `distilbert` encoder produced finite `[1, 4, 768]` output
in 0.0375 seconds with one CPU thread. This used no GPU and did not alter the
separate running Orbitune process. Exact loaded parameter count was
68,712,704; trainable count was zero.

The Hub tokenizer uses reviewed Apache-2.0 custom Python code, MeCab, and
SentencePiece with `unidic_lite`. These dependencies were installed only in a
new isolated virtual environment (`fugashi==1.3.2`,
`sentencepiece==0.2.0`, `unidic-lite==1.0.8`); the base environment was not
changed. After reviewing the pinned tokenizer code, four fixed Japanese
question/candidate pairs produced repeatable token IDs within the model
vocabulary, and the frozen encoder returned finite `[4, 22, 768]` output. This
is a local tokenizer+encoder smoke, not a comparison to an independent
tokenizer implementation. It does not establish Decision quality, calibration,
or package fit.

## Decision and next gate

**Checkpoint feasibility: PASS. Short tokenizer+encoder smoke: PASS. Decision
quality: NOT MEASURED. Product/release eligibility: NOT ESTABLISHED.**

The next gate is a small CPU-only smoke through the same typed-decision input
pipeline, followed by an explicitly limited paired readout diagnostic only if
the active CPU workload has ended. The existing typed-decision validation is
already exposed and uses synthetic teacher distributions, so any reuse must be
labelled a diagnostic rather than independent confirmation; a new human-labeled
decision task remains necessary for a quality claim.

## Sources

- [Pinned model revision](https://huggingface.co/line-corporation/line-distilbert-base-japanese/tree/c32d05753d0d4be0f60482291a9452ec7e6f7d68)
- [Model card and declared JGLUE metrics](https://huggingface.co/line-corporation/line-distilbert-base-japanese)
- [Official LINE model repository](https://github.com/line/LINE-DistilBERT-Japanese)

