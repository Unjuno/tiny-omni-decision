# SocialIQA post-exposure compact BERT comparison — 2026-11-24

## Decision

This is a **post-exposure exploratory comparison**, not an independent
publisher-development evaluation or a prospective BERT-family selection
study. A prior project run had already trained a frozen all-MiniLM-L6 readout
on this same publisher train source and inspected the official SocialIQA dev
split once. Its report explicitly barred future tuning/re-evaluation there.
The 2026-11-24 run nevertheless evaluated two different MiniLM candidates on
the same public dev split. Preserve the run and predictions for transparency,
but do not use these results for model selection, product claims, or as fresh
confirmation.

For completeness, the new frozen MiniLM-L3 and MS-MARCO MiniLM-L6 candidates
scored about 45% on that already-exposed dev split, far below the 95–98%
aspiration. Their descriptive accuracy difference was only 0.20 points, with
a context-group paired 95% interval spanning zero. A prior `all-MiniLM-L6-v2`
run scored 43.40% / 1.0739 NLL / 0.6485 Brier / 0.0763 ECE-15 on the same dev
split. These are not matched model comparisons: the encoders, input formatting,
training recipes, and internal splits differ. The internal train-derived
validation scores around 55% also reuse a training source already exposed to
prior project fitting. This report therefore supplies no new independent
generalization evidence and does not establish a BERT-family ceiling.

This text-only run does not close any image, audio, video, fusion, calibration,
or product-size gate. The SocialIQA source has no stable ATOMIC source-event IDs,
so exact context grouping cannot guarantee event-disjointness. The text-path
choice should remain task-specific. The broader BERT evidence reconciliation remains in
[`R2_BERT_FAMILY_RECONCILIATION_20261123.md`](R2_BERT_FAMILY_RECONCILIATION_20261123.md).

## Source and split integrity

The experiment used the official [Social IQa dataset card](https://huggingface.co/datasets/allenai/social_i_qa),
pinned at dataset revision `8835ceb9141d7896d9d968634a9b21ae440e3ec5`, and
AI2's [Rainbow source repository](https://github.com/allenai/rainbow), pinned
at `ddc370727bb145bbc53c65f5bd39db4beca4115d` for the loader definition. The
card declares CC BY 4.0. The fetched official train/dev archive
SHA-256 is
`ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3`; it
matches the publisher checksum sidecar. Only `train.jsonl`, `train-labels.lst`,
`dev.jsonl`, and `dev-labels.lst` were extracted. The official test split was
not downloaded or opened.

The source contains 33,410 train and 1,954 development questions. Exact
normalized-context grouping found 33,000 train context groups and 1,930 dev
context groups, with zero normalized context overlap between publisher splits.
Seed 17 assigned entire train context groups to internal fit/selection:

| Partition | Questions | Context groups |
|---|---:|---:|
| Internal train | 30,069 | 29,692 |
| Internal validation | 3,341 | 3,308 |
| Publisher development, evaluated after selection | 1,954 | 1,930 |

Train/internal-validation context overlap and publisher train/dev context
overlap were both zero. The internal validation ID-order hash is recorded in
the external split manifest. No questions or candidate answers were removed.
However, this 2026-11-24 split was made from the same publisher training source
that had already been used in the prior SocialIQA probe; these rows are not a
new independent validation cohort. Exact context grouping also cannot rule
out shared paraphrased ATOMIC events.

## Frozen model and scoring protocol

Both candidates use the same English BERT-family input and scorer. For each
option, the encoder receives `[context + question, candidate answer]`; its
final-layer CLS vector is passed to a shared `hidden -> 128 tanh -> 1` scorer.
The three candidate scores are softmaxed into JSON-compatible probabilities.
Option scoring is shared across positions, so candidate permutation permutes
scores without changing them. Only the 49,409-parameter scorer was trained;
all encoder parameters were frozen. This is not a pure layer-depth comparison:
the L3 checkpoint was sentence-paraphrase trained, while L6 was MS MARCO
cross-encoder trained. The comparison tests two usable BERT-family candidates
under a matched downstream protocol, not the causal effect of adding layers.

| Candidate | Pinned revision | Encoder parameters | Serialized encoder weights | Best internal epoch |
|---|---|---:|---:|---:|
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | `4ca70771034acceecb2e72475f72050fcdde4ddc` | 17,389,824 | 69,569,488 bytes, SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` | 8 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 22,713,216 | 90,870,598 bytes, SHA-256 `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae` | 14 |

Including the scorer, the respective total resident parameter counts are
17,439,233 and 22,762,625. The BERT checkpoint itself is under the provisional
200 MB package target, but this is only a text encoder plus a small scorer, not
the integrated multimodal product.

Seed 17, maximum 20 epochs, early-stopping patience 3 and minimum validation
NLL improvement `1e-4` were fixed. AdamW used learning rate `1e-3`, weight
decay `1e-4`, question batch size 96. Maximum pair length was 256 tokens;
measured truncation was zero for all train, internal validation, and publisher
development candidate pairs. The selected checkpoint for each encoder was
chosen by lowest internal-validation NLL; no publisher-development metric was
used for checkpoint or encoder selection. The L3 run stopped after epoch 11
with epoch 8 selected. L6 stopped after epoch 17 with epoch 14 selected.

## Results

### Internal validation used for selection

| Candidate | Accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|
| MiniLM-L3 | 55.37% | 0.9330 | 0.5530 | 0.0250 |
| MiniLM-L6 | 55.37% | 0.9449 | 0.5612 | 0.0221 |

Within this exploratory run, MiniLM-L3 had the lower NLL under the fixed rule.
Because the source and its earlier validation results were already exposed,
this does not qualify as a prospective model selection.

### Repeated official publisher-development evaluation (exploratory only)

The split was already evaluated in the prior project audit. Repeating it here
was a process error. Keep these numbers for transparency, but exclude them from
future tuning and treat the bootstrap below as descriptive only; it does not
restore independence or correct for repeated exposure.

| Candidate | Accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|
| MiniLM-L3 | 45.04% | 1.0802 | 0.6479 | 0.0953 |
| MiniLM-L6 | 44.83% | 1.0896 | 0.6564 | 0.0821 |
| Uniform three-choice reference | 33.33% | 1.0986 | 0.6667 | 0.0000 |
| Train class-prior probabilities | 32.91% | 1.0988 | 0.6668 | 0.0084 |

Paired bootstrap over the 1,930 normalized context groups, 10,000 resamples,
seed 1729, gave L3 minus MS-MARCO-L6 accuracy `+0.20 pp`, 95% percentile
interval `[-2.56, +2.83] pp`. Since this repeats an exposed public split and
the bootstrap is conditional on these candidates and procedures, it is not
fresh confirmatory evidence. The internal-to-publisher score gap is
descriptive, not a clean training-overfit estimate; the training records,
representations, and protocols differ from the prior readout study.

## Reproduction and artifacts

The pinned command was:

```powershell
python -m scripts.pretrained_reuse.run_socialiqa_minilm_probe `
  --data-dir C:\CodexArtifacts\pretrained-reuse-r2-local\socialiqa-source-screen-20261124-attempt-01\source-payload `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2-local\socialiqa-minilm-decision-20261124-attempt-01 `
  --hf-cache C:\CodexArtifacts\HuggingFace-cache-MIDI\hub `
  --seed 17 --max-length 256 --encode-batch-size 96 `
  --question-batch-size 96 --max-epochs 20 --patience 3
```

Environment: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA available, Transformers
5.6.2, safetensors 0.8.0, NVIDIA GeForce RTX 3080 Laptop GPU (16 GB). GPU was
idle before the run; sampled allocation during L6 execution was 853 MiB, but
peak memory was not instrumented and is therefore UNKNOWN. Per-candidate
feature-extraction plus fit wall times recorded by the script were 33.32 s
(L3) and 63.24 s (L6); these exclude the subsequent publisher-development
forward pass. The full command completed successfully and the GPU returned to
0 MiB allocated. No cloud compute was used.

The external run directory contains effective config, split manifest, training
history, internal-validation and publisher-development predictions, readout
checkpoints, metrics, and hashes. Important source file hashes:

| File | SHA-256 |
|---|---|
| `train.jsonl` | `4c5333dea02be6d212114a83f17f7bbcbdb53b3d692f476bd40dcd242b6233de` |
| `train-labels.lst` | `37cd494b60750f4f7a8c7cdd15fad545f4159070dbeb0431597e06e996ae2fd6` |
| `dev.jsonl` | `d8128dfe60783a9971b954ca82dd57b87b05d5d40468876d51e8a1ad2c557465` |
| `dev-labels.lst` | `2d882ff71969d3980d37dd07aecdf9093c6e63e27c670f3a203bcf3b9b74bc05` |

The complete machine-readable source record is
[`socialiqa-text-decision-probe-20261124.json`](../../manifests/pretrained_reuse/socialiqa-text-decision-probe-20261124.json).
Model weights, source media/data and run outputs are outside Git; only code,
tests, the compact manifest and this report are committed.

## Recommendation

This repeated run does **not** justify another BERT decision or a depth sweep.
The prior SocialIQA audit itself had already concluded that the next BERT
comparison must use a newly admitted source-disjoint task. Do not reopen
SocialIQA dev or use these repeated metrics for selection. The next
product-level work should prioritize admitted multi-input tasks,
modality-necessity controls, and calibrated fusion. This run does not justify
claiming any 95–98% modality gate met.

Focused CPU tests: 5 passed; full repository tests: 2,901 passed / 11 skipped;
Ruff, manifest validation, and the post-fix GitHub Actions run passed. The
official test and project sealed audit were not accessed. The official public
dev split was accessed again despite its previous exposure; the results above
are explicitly marked exploratory and invalid as independent confirmation.
