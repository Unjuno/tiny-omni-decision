# Japanese Ruri-v3-70m option-scoring probe (2026-11-23)

## Decision

The frozen Ruri-v3-70m representation with the same shared option scorer
improved the selected development-validation point estimates over frozen
Ruri-v3-30m on this one Japanese five-choice task. This is a bounded reason to
keep Japanese BERT-family encoders in the text-path candidate set. It is not
evidence of 95% quality, general Japanese reasoning, multimodal quality, or an
independent test result. Do not tune this already exposed validation again.

## Source and fixed split

The experiment reused the existing JGLUE v1.1.0 JCommonsenseQA development
corpus and exact-question-grouped seed-17 split described in the
[30m probe](R2_JCOMMONSENSEQA_RURI_READOUT_20261122.md). The source is CC BY-SA
4.0. It has 8,054 training rows / 7,938 normalized question groups and 2,004
validation rows / 1,984 groups; normalized question-group overlap is zero.
All rows were retained, including repeated groups with conflicting answers.
The pinned JGLUE release contains no test payload, and no official test was
downloaded or opened. The same validation was used for NLL checkpoint
selection and model comparison, so all metrics below are development-only.

## Candidate and execution

The encoder was [`cl-nagoya/ruri-v3-70m`](https://huggingface.co/cl-nagoya/ruri-v3-70m)
at revision `07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc`; its model card declares
Apache-2.0. The local safetensors file is 280,019,040 bytes with SHA-256
`c959b76a535eee838b1f1e6b8999ba513ef8d87a6f8c3534c50dcda1eb5f6a62`. The
loaded `ModernBertModel` has 13 layers, hidden size 384 and exactly 70,002,816
parameters. Mean-token pooling plus L2 normalization was used for each
question and option.

The scorer stayed the same functional form as the 30m reference:
shared `4d -> 128 -> 1` MLP over `[q,c,abs(q-c),q*c]`, applied separately to
each of five choices. The hidden width, optimizer, split, seed, eight-epoch
budget, batch size, feature pipeline, and minimum-validation-NLL selector were
held fixed. The wider encoder changes scorer input width and therefore its
size: 196,865 trainable scorer parameters versus 131,329 for the 30m run.
Total pre-quantization parameters are 70,199,681. No encoder parameters were
updated. This fixed readout probe compares frozen representations; it is not
an end-to-end BERT fine-tuning comparison.

The run used seed 17, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
and the local Intel Core i7-12700H CPU with four PyTorch threads. It ran from
`2026-11-22T15:24:05.751381Z` to `2026-11-22T15:29:11.641640Z` (305.89 s):
299.66 s feature extraction and 4.33 s scorer training. GPU and cloud compute
were not used. The 280 MB FP32 encoder weight file alone exceeds the
provisional 200 MB package target; no quantization or package integration was
performed.

## Results

| Frozen encoder / selected epoch | Encoder params | Readout params | Total params | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ruri-v3-30m / 8 | 36,705,536 | 131,329 | 36,836,865 | 47.65% | 1.29043 | 0.64738 | 0.03889 |
| Ruri-v3-70m / 7 | 70,002,816 | 196,865 | 70,199,681 | **50.70%** | **1.21925** | **0.61291** | **0.02917** |

Candidate-minus-30m paired bootstrap over the same 1,984 normalized question
groups, 10,000 replicates, seed 17:

| Metric | Difference | 95% percentile interval |
|---|---:|---:|
| Accuracy | +3.04 percentage points | +0.80 to +5.26 pp |
| NLL | -0.07118 | -0.09829 to -0.04343 |
| Brier | -0.03446 | -0.04744 to -0.02131 |

These intervals are descriptive only: both candidates' checkpoints were
selected on this validation, and the same validation was used for comparison.
They are not independent confirmation or a blind audit. ECE-15 improved by
0.00973 in point estimate; no ECE confidence interval was computed.

At selected epoch 7, train Accuracy was 52.48%, a 1.79-point train/validation
gap. Training CE continued to decline through epoch 8. Validation NLL improved
slowly through epoch 7 and rose slightly at epoch 8 (1.21925 to 1.22070),
while Accuracy fluctuated. This shows a modest late NLL reversal, not evidence
of a broad capacity or optimization ceiling.

The saved readout was loaded from disk and replayed on cached validation
features: argmax matched on 2,004/2,004 rows and maximum probability delta was
0.0. Official test access remained false. The complete report, frozen
features, checkpoint, predictions, reload receipt, paired comparison, and
executed training-script snapshot are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\jcommonsenseqa-ruri70m-readout-v1-attempt-01`.
Their hashes and source pins are in
[`jcommonsenseqa-ruri-readout-20261122.json`](../../manifests/pretrained_reuse/jcommonsenseqa-ruri-readout-20261122.json)
and the run's `run-provenance.json`.

## Interpretation and next gate

The 70m frozen encoder is a better text-path candidate than the 30m encoder
for this development probe, with a small-to-moderate absolute gain and much
lower five-choice error than uniform chance. At 50.70% Accuracy, it remains
far from the project's 95% aspiration. The result does not show whether the
gap is due to task difficulty, pretrained knowledge, frozen features, scorer
capacity, or annotation ambiguity. The grouped split prevents exact question
overlap but does not establish independent source-author or knowledge groups.

Do not continue selecting models or adjusting readouts on this exposed split.
The next useful Japanese BERT-family step requires a separately admitted,
rights-cleared Japanese Decision task with independent groups and an untouched
test, or must be explicitly labeled exploratory if no such source is
available. English fixed-label intent results remain a separate, narrow text
use case. The four-modality Decision objective, genuine paired-input
complementarity, and compact quantized package remain open.
