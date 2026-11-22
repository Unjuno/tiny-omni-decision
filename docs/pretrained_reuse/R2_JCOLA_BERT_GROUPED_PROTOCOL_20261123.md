# JCoLA BERT-family grouped Decision comparison — 2026-11-23

## Purpose and status

This is a new, task-specific text Decision diagnostic for a frozen Japanese
BERT teacher and a parameter-budgeted BERT-family candidate. JCoLA asks for a
binary linguistic-acceptability judgment; the adapter expresses it as
probabilities over two supplied options. This narrow task cannot establish
general text, multimodal, or 95–98% product quality.

Status at protocol freeze: source and split audited; code and CPU smoke passed;
full model comparison is **not yet run**. No publisher test or sealed audit is
part of the experiment.

## Source and split integrity

Source: [`osekilab/JCoLA`](https://github.com/osekilab/JCoLA), pinned to
`736d9eef3af04bb17e4cf59adf01325973234ff9`, declared CC BY-SA 4.0. Local files
are retained under `C:\CodexArtifacts\pretrained-reuse-r2-local\jcola-japanese-acceptability-20261123-attempt-01`;
raw data and model weights are not added to Git.

The selected source records are 6,919 in-domain training sentences and 685
official out-of-domain development sentences. The in-domain publisher
validation file is excluded: it shares all ten source/article groups with
in-domain training. Instead, two source groups were held out from in-domain
training using Python `random.Random(17)` after sorting source IDs:

| Partition | Examples | Number of source groups | Source groups |
|---|---:|---:|---|
| Readout training | 6,342 | 8 | All except `Kuroda_1965`, `Shibatani_1990` |
| Checkpoint-selection validation | 577 | 2 | `Kuroda_1965`, `Shibatani_1990` |
| Official out-of-domain development | 685 | 27 | Separate published out-of-domain sources |

The 27 out-of-domain sources have zero source overlap with all in-domain
training sources. Normalized sentence overlap is zero for train/selection
validation and for train/out-of-domain development. The source-grouped
assignment SHA-256 is
`a537429c5a06ff78612b722c7359de7770236b7fcd1eac96b9bde0899108aa7c`.
Train JSON SHA-256 is
`232787aa7bb11f3c42e455aad59d5f54cf1ea8255e9de79f77840de60e0541bf`; official
out-of-domain development JSON SHA-256 is
`4bbb109d42bbceb52ab139b30890be871755672749782abe2e034903b7662930`.

The official out-of-domain development set is public and is used once after
both checkpoints are selected using only the two held-out training sources.
It is development evidence, not a sealed or blind audit. The publisher's test
records are not included in the selected source checkout and are not loaded.

## Frozen candidates and decision contract

Both encoders remain frozen. The only trained parameters are a shared
`768 -> 128 tanh -> 1` option scorer. For each sentence, it assigns scores to
`["不適格（文法的に許容されない）", "適格（文法的に許容される）"]`, then
softmax produces the returned option probabilities. A deterministic option
swap check verifies that reversing option order reverses score order without
changing the underlying decision.

| Candidate | Encoder parameters | Readout parameters | Text-only total before quantization | Role |
|---|---:|---:|---:|---|
| Tohoku BERT-base v3 | 110,616,576 | 98,561 | 110,715,137 | Frozen teacher/reference; exceeds current four-tower budget when substituted into the full stack |
| LINE DistilBERT Japanese, first 4 of 6 layers | 53,912,064 | 98,561 | 54,010,625 | Frozen size-constrained candidate; the upper two pretrained layers are omitted |

The 4-layer model uses only pretrained layers 0–3 from pinned LINE revision
`c32d05753d0d4be0f60482291a9452ec7e6f7d68`. Its four-tower arithmetic fits
within the provisional 200M pre-quantization encoder budget, but full-system
quality and resident-head costs still need to be checked. The six-layer
checkpoint is not substituted unchanged because the current stack would be
about 210.17M before the product Decision head.

## Frozen fit and selection policy

- Seed 17; maximum 20 epochs; stop after three epochs without improvement.
- Select the readout checkpoint by source-held-out validation NLL only.
- AdamW, learning rate `0.002`, weight decay `0.0001`; batch size 64 decisions.
- Encode max length 512, candidate-pair batch 16; frozen encoder and FP32
  features/readout. No encoder fine-tuning, quantization, calibration fitting,
  prompt search, or sealed-audit evaluation.
- The official out-of-domain development examples are not passed to either
  model until both readout checkpoints and selection epochs have been saved.

Report Accuracy, macro Accuracy, MCC, NLL, Brier, ECE-15, mean confidence,
per-class recall, per-source metrics, training curves, truncation counts,
parameter counts, environment, and runtime. Save each out-of-domain
development prediction with UID, source, target, predicted option, and both
option probabilities. Treat results as one public development comparison,
not a product promotion.

## Implementation and smoke evidence

The focused script is `scripts/compare_jcola_bert_decision.py`; source-group,
content-overlap, determinism, and option-swap tests are in
`tests/test_jcola_decision_experiment.py`. The CPU-only smoke loaded both
models, verified the exact first-four-layer checkpoint load, kept all encoder
parameters frozen, fit a tiny readout, and did not load or score the official
out-of-domain development file. Its metrics are plumbing-only.

Run command for the full fixed comparison (output path must be new):

```powershell
$env:PYTHONPATH = 'src'
& 'C:\CodexArtifacts\pretrained-reuse-r2-local\line-distilbert-tokenizer-venv-20261123-attempt-01\Scripts\python.exe' scripts/compare_jcola_bert_decision.py `
  --source-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\jcola-japanese-acceptability-20261123-attempt-01' `
  --bert-base-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\jnli-tohoku-bertbasev3-model-20261123-attempt-01' `
  --line-distilbert-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\line-distilbert-ja-20261123-attempt-01' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\jcola-bert-grouped-decision-20261123-attempt-01' `
  --device cuda --cpu-threads 1 --seed 17 --epochs 20 --patience 3 `
  --batch-size 64 --encode-batch-size 16 --max-length 512
```

Full comparison results, artifact hashes, and execution status will be added
without rewriting this preregistered protocol.
