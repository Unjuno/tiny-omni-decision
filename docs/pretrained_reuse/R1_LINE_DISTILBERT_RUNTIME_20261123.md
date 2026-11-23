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
SHA-256 is `d4dfa9a6b1fbfd4c9c32deecf7ea1985f1ca884a030d8086a71552060f7c2529`.

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

## Typed-Decision pipeline smoke

A small CPU-only smoke passed through the repository's actual frozen-feature
and candidate-readout pipeline. It used seed 17, one state per workflow in each
partition (four grouped states / 20 decisions for train and validation), two
readout epochs, one CPU thread, and the same frozen synthetic teacher-label
source. The publisher's public test was not loaded. The tokenizer had 32,000
IDs while the model's embedding table has 32,768 rows; the 768 extra rows are
unused padding, and every emitted ID was checked against the embedding table.

The frozen encoder has **68,087,808 parameters**; the readout has **98,561**,
for **68,186,369 pre-quantization parameters** in this smoke configuration.
The full upstream masked-LM checkpoint has 68,712,704 parameters, but its MLM
head is omitted from Decision inference. Feature extraction took 51.61 seconds
for 142 candidate pairs on CPU; the readout fit took 0.014 seconds. The best
readout was epoch 2, and reloading reproduced all probabilities exactly.

The tiny validation smoke produced macro teacher-agreement Accuracy 0.375,
soft NLL 1.1762, soft Brier 0.1624, and ECE 0.1917. With only 20 decisions and
checkpoint selection on those same rows, these figures verify plumbing only;
they are **not** a model-quality estimate and must not be compared to the full
BERT-base / LightSPLADE run or any 95% product target. The full paired
comparison remains unrun. The smoke output and hashes are preserved outside
Git in `typed-decisions-ja-line-distilbert-smoke-20261123-attempt-03`, with
manifest SHA-256
`21f575044f3e27739d0c9eec403295659019ac4c035c96e502f94d624ff6389e`.

This smoke exposed integration issues before the successful attempt:
`AutoModel` passes no `add_pooling_layer` argument to DistilBERT, the custom
tokenizer needs its reviewed local code explicitly enabled, and its 32,000-token
SentencePiece vocabulary is padded to a 32,768-row embedding. The comparison
CLI now has an opt-in local tokenizer-code flag, falls back to the model's
native no-pooler constructor, and validates emitted IDs against the actual
embedding bounds. Pure CPU tests cover padded vocabularies and invalid IDs.
The implementation is commit `71fb2deef77727abeb5e61f8b408873f767a9179`.

## Decision and next gate

**Checkpoint feasibility: PASS. Short tokenizer+encoder smoke: PASS. Decision
quality: NOT MEASURED. Product/release eligibility: NOT ESTABLISHED.**

### Four-tower parameter fit

To check the 200M pre-quantization ceiling, the encoder was also instantiated
from the pinned config with fewer DistilBERT layers and counted directly; no
weights were truncated or evaluated in these counts. Replacing the current
36,705,536-parameter Ruri text tower in the 178,689,152-parameter four-tower
proposal gives:

| LINE DistilBERT encoder depth | Encoder parameters | Four-tower total with the 98,561-parameter smoke readout | Margin to 200M |
|---:|---:|---:|---:|
| 6 (pinned checkpoint) | 68,087,808 | 210,169,985 | -10,169,985 |
| 4 (config-only count) | 53,912,064 | 195,994,241 | +4,005,759 |
| 3 (config-only count) | 46,824,192 | 188,906,369 | +11,093,631 |

The 3-layer row remains a config-only arithmetic check. A CPU inheritance
smoke now confirms that the four-layer candidate can actually load the first
four layers from the pinned six-layer checkpoint with no missing model keys.
It has 53,912,064 encoder parameters, zero trainable parameters in the frozen
smoke, and emits finite `[1, 4, 768]` hidden states. Transformers reports 38
expected unexpected keys: weights for discarded source layers 4–5 and the
masked-language-model head, which is not part of the Decision encoder. This
proves load/forward feasibility only; it does not show that truncating the
upper layers preserves decision quality. The six-layer checkpoint therefore
does not fit the present four-tower budget unchanged; the four-layer variant
still requires a fixed-protocol quality and runtime evaluation, with all
resident heads included before acceptance.

### Four-layer inheritance smoke — 2026-11-23

The pinned six-layer checkpoint was loaded locally and offline with
`AutoModel.from_pretrained` after setting `DistilBertConfig.n_layers = 4`.
PyTorch reported no missing keys. The 38 extra checkpoint keys were confined
to source layers 4–5 and the MLM prediction head. After loading, all encoder
parameters were frozen; a tokenizer-independent four-token forward returned a
finite `[1, 4, 768]` tensor. The encoder count was 53,912,064; trainable count
was zero. On Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, and one
CPU thread, load took 1.269 s and the smoke forward took 0.0832 s. No GPU,
tokenizer, real text, gradient, optimizer, fit, or quality evaluation was used.

The source weight file was 274,894,944 bytes with SHA-256
`9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
The authoritative external receipt is
`C:\CodexArtifacts\pretrained-reuse-r2-local\line-distilbert-4layer-load-smoke-20261123-attempt-03\smoke.json`,
SHA-256 `7c8b1223f5d5887f445810ea6e901438891b7f02f2fa19b1be43a5549d86214d`.
Attempt 01's forward was finite, but its receipt incorrectly counted
`requires_grad` flags as trainable even though no optimization occurred; it is
superseded (receipt SHA-256
`90d49beea36e4e88d7b579c2b20be547ac9ae85d75dfd6aa240e167127154494`). Attempt
02 completed the freeze and model checks but failed only while serializing a
set-valued field; its incomplete-status receipt is preserved with SHA-256
`b01b7442ae2ab25afb54a3d4efc3c252df94bf44ccff90b3ebec1d86cebc3eb2`.

The next gate is an explicitly limited paired readout diagnostic only after
the active CPU workload has ended. The existing typed-decision validation is
already exposed and uses synthetic teacher distributions, so any reuse must be
labelled a diagnostic rather than independent confirmation; a new human-labeled
decision task remains necessary for a quality claim.

## Sources

- [Pinned model revision](https://huggingface.co/line-corporation/line-distilbert-base-japanese/tree/c32d05753d0d4be0f60482291a9452ec7e6f7d68)
- [Model card and declared JGLUE metrics](https://huggingface.co/line-corporation/line-distilbert-base-japanese)
- [Official LINE model repository](https://github.com/line/LINE-DistilBERT-Japanese)

