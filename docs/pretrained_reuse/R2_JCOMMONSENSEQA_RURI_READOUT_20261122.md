# R2 Japanese five-choice BERT-family probe (2026-11-22)

## Purpose and decision

This probe asks whether a small Japanese pretrained encoder can support a
direct five-choice Decision task. It is a useful follow-up to the English
intent-classification checks, but it is one Japanese commonsense-QA task and
does not establish general text reasoning, multimodal fusion, or the product
quality target. The result supports keeping the BERT-family route open for
text while showing that this frozen-encoder candidate is not yet good enough
for the product target.

## Source, rights, and split

The data came from the public [JGLUE v1.1.0 release](https://github.com/yahoojapan/JGLUE/tree/v1.1.0),
pinned to commit `6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33`. Its license is
CC BY-SA 4.0. The source has 8,939 publisher-train rows and 1,119 publisher-
validation rows. The release contains no test payload; no test data was
downloaded or opened. Product redistribution and derivative-model use remain
subject to rights review; this experiment is not a license clearance.

The two published development splits were combined and repartitioned by
normalized question identity (NFKC, casefold, collapsed whitespace), using
seed 17. All rows were retained, and each exact normalized question group
stayed entirely on one side. There are 9,922 unique question groups, 102
repeated-question groups, and 69 repeated groups with conflicting correct
answer text. One row has duplicate choice text; no target text is duplicated
within its choices. No rows were removed or rewritten.

| Development partition | Rows | Unique question groups | Publisher train rows | Publisher validation rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,054 | 7,938 | 7,171 | 883 |
| Validation | 2,004 | 1,984 | 1,768 | 236 |

Question-group overlap is zero. This guards exact question repetition, but not
semantic paraphrase, source-author, or knowledge overlap. Checkpoint selection
and candidate comparisons both use this validation set; all results are
development evidence. The official test is unavailable in the pinned release.

## Fixed candidate

The encoder was `cl-nagoya/ruri-v3-30m`, pinned at
`24899e5de370b56d179604a007c0d727bf144504` (Apache-2.0 model card). It has
36,705,536 frozen parameters. Mean-token pooling followed by L2 normalization
produced question and option vectors. A shared candidate scorer over
`[question, option, abs(question-option), question*option]` had 131,329
trainable parameters. The combined encoder and scorer therefore contain
36,836,865 pre-quantization parameters. The scorer is order-equivariant across
choices: it applies the same score function independently to each option,
then softmaxes the five scores.

The run used seed 17, CPU FP32, AdamW with LR 0.001 and weight decay 0.0001,
batch size 64 questions, and eight full epochs. The Ruri encoder stayed frozen.
Validation NLL selected epoch 8. The run started at `2026-11-22T14:43:16Z` and
ended at `2026-11-22T14:45:57Z`, without early stopping or GPU use. Feature
extraction took 153.57 seconds and readout training took 4.81 seconds. The
machine CPU was Intel Core i7-12700H (14 cores / 20 logical processors).
Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2.

Exact run command:

```powershell
python scripts/train_jcommonsenseqa_ruri_readout.py `
  --train C:\CodexArtifacts\pretrained-reuse-r2-local\jcommonsenseqa-grouped-split-v1-attempt-01\train.jsonl `
  --validation C:\CodexArtifacts\pretrained-reuse-r2-local\jcommonsenseqa-grouped-split-v1-attempt-01\validation.jsonl `
  --split-manifest C:\CodexArtifacts\pretrained-reuse-r2-local\jcommonsenseqa-grouped-split-v1-attempt-01\split-manifest.json `
  --model C:\CodexArtifacts\pretrained-reuse-r1\ruri-v3-30m `
  --revision 24899e5de370b56d179604a007c0d727bf144504 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2-local\jcommonsenseqa-ruri-readout-v1-attempt-01 `
  --seed 17 --epochs 8
```

## Results

| Model / rule | Accuracy | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: |
| Uniform five-choice prediction (expected Accuracy) | 20.00% | 1.60944 | 0.80000 | 0.00000 |
| Train-majority answer position (choice 1) | 20.41% | 15.91816 | 1.59182 | 0.79591 |
| Frozen Ruri cosine similarity | 45.51% | 1.58945 | 0.79188 | 0.24702 |
| Frozen Ruri + learned shared scorer, selected epoch 8 | **47.65%** | **1.29043** | **0.64738** | **0.03889** |

The learned scorer improved on raw cosine by 2.15 percentage points Accuracy
and substantially improved probability quality. On the selected epoch,
training Accuracy was 48.49%, versus 47.65% validation Accuracy, a 0.83-point
gap. Validation NLL improved through all eight epochs; Accuracy fluctuated
around 47–48%, with epoch 7 at 47.90%. There is not a strong late overfitting
signal in this short run, but eight epochs do not prove a plateau. ECE rose
from 0.0291 at epoch 7 to 0.0389 at the NLL-selected epoch 8, so the selector
traded a small calibration change for lower NLL. No calibration fitting was
performed. A separate reload comparison matched argmax on all 2,004 validation
rows with maximum probability difference `0.0`.

The absolute result remains far below the aspirational 95–98% target. This
task is a hard, knowledge-dependent five-choice benchmark; one seed and one
development split cannot distinguish data ceiling, pretrained representation,
task adaptation, or benchmark ambiguity. The experiment is evidence to test
Japanese BERT encoders further, not evidence to claim BERT has solved the
Decision objective.

### Frozen joint question-option representation follow-up (2026-11-23 JST)

To test whether independently embedded question and option vectors discarded
useful interaction information, a second candidate passed every
question-option pair together through the same frozen Ruri encoder. Only the
shared scorer changed; it used a 256-to-509-to-1 MLP (131,323 trainable
parameters), seed 17, the same train/validation files, AdamW schedule and
eight-epoch cap. This was frozen feature extraction, not Ruri fine-tuning.

| Candidate | Accuracy | NLL | Brier | ECE-15 | Selected epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Separate question/option features + shared scorer | 47.65% | 1.29043 | 0.64738 | 0.03889 | 8 |
| Joint question-option encoding + shared scorer | 42.91% | 1.36543 | 0.69143 | 0.04126 | 7 |

The paired difference was computed on the same 2,004 rows; a 10,000-replicate
bootstrap resampled 1,984 normalized exact-question groups (seed 17). Joint
encoding minus separate encoding was `-4.74 pp` Accuracy (percentile 95% CI
`[-7.05, -2.45]`), `+0.0750` NLL (`[+0.0500, +0.1005]`) and `+0.0441` Brier
(`[+0.0316, +0.0571]`). The checkpoint was selected on this same validation,
so these intervals are descriptive after selection, not independent
confirmation. This tested mean pooling of a frozen joint pair; it does not
rule out fine-tuning, task prefixes, other pooling, or stronger Japanese
encoders. However, it provides no reason to continue tuning this already-used
split. Any next Japanese option-scoring experiment needs a newly admitted,
independent source/task or an explicit label as exploratory development work.

Joint-pair features were extracted on the local RTX 3080 Laptop GPU in 28.86 s,
with 314,831,872 bytes peak allocated; fitting the readout took 4.09 s. The
encoder remained frozen. Exact settings, the full curve and artifacts are in
the paired candidate's external run report and the machine manifest below.

## Reproduction artifacts

The full prepared data, cached frozen features, checkpoint, validation
probabilities, and run metadata are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\`:

- `jcommonsenseqa-grouped-split-v1-attempt-01`
- `jcommonsenseqa-ruri-readout-v1-attempt-01`
- `jcommonsenseqa-ruri-readout-analysis-attempt-03`
- `jcommonsenseqa-ruri-pair-readout-v1-attempt-01`
- `jcommonsenseqa-ruri-readout-paired-analysis-attempt-01`

The cached features are 208,792,234 bytes; the readout checkpoint is 527,352
bytes. Key hashes are recorded in
[`jcommonsenseqa-ruri-readout-20261122.json`](../../manifests/pretrained_reuse/jcommonsenseqa-ruri-readout-20261122.json).
The existing Ruri model and earlier experiments were not changed.

## Recommended next BERT-family work

The most informative next test is a small same-split, same-readout comparison
against one additional Japanese encoder, chosen by explicit license, size,
and local availability checks. Keep the source rows and grouped split fixed.
If this task is to select a candidate for deployment, then separately test a
task-adapted encoder or a compact cross-encoder under a predeclared validation
selector; do not describe that as a general BERT-family result. A genuinely
blind external evaluation cannot be done on this JGLUE release because its
test payload is absent. Before product use, obtain an independently sourced
Japanese task with suitable rights and a held-out source grouping. Multimodal
and R2-M requirements remain open regardless of this text result.
