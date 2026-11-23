# R2 multilingual BERT-family probe — XCOPA validation

## Decision

Keep BERT-family models in the research plan as text specialists and possible teachers. This
single frozen multilingual MiniLM probe does not justify adding the model to the multimodal runtime
or claiming that BERT solves text decision quality. Its accuracy is above the two-choice chance
level, while its probability quality is worse than a uniform predictor. It is also a semantic
similarity baseline rather than a task-aware causal reasoner.

## Fixed scope

- Dataset: [XCOPA](https://github.com/cambridgeltl/xcopa), repository revision
  `e2e9d7f105a758ee869cd13e0fe251ef93a29840`, repository license CC BY 4.0.
- Only the official `data/*/val.*.jsonl` split was fetched. The benchmark documents 11 languages;
  the pinned model card supports the six directly tagged languages in this run plus Mandarin via
  its `zh-cn` / `zh-tw` tags. Haitian Creole, Quechua, Swahili, and Tamil were excluded because the
  model card does not list them.
- Seven languages × 100 records = 700 validation examples. No training split, test split, or sealed
  audit was opened. No model training, calibration, or checkpoint selection was performed.
- Model: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
  revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, Apache-2.0. The model card describes a
  `BertModel` with masked mean pooling and a 128-token sequence limit. The loaded model has
  **117,653,760 parameters**; the safetensors file is 470,641,600 bytes.
- Frozen scoring rule: cosine similarity between the masked-mean embedding of the premise and each
  choice, divided by temperature `0.05`; the larger score wins. No prompt search or validation
  threshold selection was performed. The scorer does not consume the `cause` / `effect` question
  field, so results measure generic premise-choice similarity, not causal-direction reasoning.

## Results

| Language | n | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Estonian | 100 | 60.0% | 1.1630 | 0.6078 | 0.2709 |
| Indonesian | 100 | 64.0% | 1.3212 | 0.5794 | 0.2484 |
| Italian | 100 | 56.0% | 1.3599 | 0.6491 | 0.2726 |
| Thai | 100 | 54.0% | 1.2690 | 0.6668 | 0.3031 |
| Turkish | 100 | 60.0% | 1.3014 | 0.6557 | 0.2892 |
| Vietnamese | 100 | 54.0% | 1.1673 | 0.6792 | 0.3006 |
| Mandarin Chinese | 100 | 68.0% | 0.9839 | 0.5033 | 0.1823 |
| **Macro across languages** | **700** | **59.4%** | **1.2237** | **0.6202** | **0.2667** |

The uniform two-choice reference is 50% expected accuracy, `ln(2) = 0.6931` NLL, and 0.5 Brier.
The model beats chance in Accuracy but is worse than uniform on NLL and Brier. Across the 700 saved
predictions, mean maximum probability is 83.65% versus 59.43% Accuracy. This is strong
overconfidence under the frozen temperature; the probabilities are not suitable as calibrated
Decision outputs. Temperature was not adjusted after inspecting validation results.

Thai and Turkish validation each contain zero `cause` questions, so those subgroup metrics are
recorded as null. The language-level validation set is small, so individual language differences
are diagnostic, not precise estimates of population performance.

## Runtime and provenance

- Local GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB.
- Python 3.11.9; PyTorch 2.6.0+cu124; Transformers 5.6.2; huggingface-hub 1.29.0;
  safetensors 0.8.0; PyYAML 6.0.3.
- Measured inference time: 0.805 seconds for the 700 rows; peak CUDA allocation: 505,284,608 bytes.
- Pinned source data Git blob IDs and SHA-256 values, model file hash, config hash, and script hash
  are in [the machine-readable manifest](../../manifests/pretrained_reuse/xcopa-minilm-zero-shot-20261124.json).
- Executed command:
  `python scripts/evaluate_xcopa_minilm_zero_shot.py --output-dir C:\\CodexArtifacts\\xcopa-minilm-zero-shot-20261124-attempt-02\\results`
- The first execution had valid predictions but serialized zero-support question-type metrics as
  non-standard NaN. A guard was added and the same pinned, unchanged scoring configuration was
  rerun to attempt 02; no model or temperature selection occurred. The final report parses as
  strict JSON.
- The evaluator and its tests were introduced at source commit
  `4dd181712cb931ed5b573fadcd7020be1c3d1177`. The corrected attempt-02 report SHA-256 is
  `1deb2368347324c1ed6e76e2a8617938e864f1faef024ec541c6f5d912f7af56`.
- Model weights, validation data, and per-example predictions remain outside Git under
  `C:\CodexArtifacts\xcopa-minilm-zero-shot-20261124-attempt-01` and
  `C:\CodexArtifacts\xcopa-minilm-zero-shot-20261124-attempt-02`.

## Implication for the broader model

This supports continued **targeted** BERT-family investigation, not a broad architecture sweep.
Existing Japanese task probes already show that larger BERT-base models can beat compact encoders
on some narrow text tasks, but they are task-dependent and do not establish the 95–98% product
goal. This multilingual MiniLM result reinforces that a strong sentence embedding is not by itself
a calibrated option Decision model.

At 117.65M parameters, this checkpoint should not be added alongside the existing roughly
170.5M-parameter Path A assembly under the provisional 200M pre-quantization budget. A later
text-tower replacement or teacher-distillation experiment would need its own matched task,
source-disjoint split, quality evaluation, and whole-stack parameter/runtime accounting. Do not
reuse this already observed XCOPA validation for selection.
