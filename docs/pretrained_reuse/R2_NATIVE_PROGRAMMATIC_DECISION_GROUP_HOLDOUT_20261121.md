# R2 programmatic decision state-group holdout

## Preregistered scope

This is a separate, TRAIN-only generalization probe on
`text:programmatic_decision_v1`. It follows the completed condition-relevance
group holdout but does not reuse that holdout for fitting or selection. The
source contains 8,192 generated decision rows across eight tasks and 1,024
underlying state groups. Each group contributes one row to each task.

The fixed 2,048-update multimodal sampler previously touched 511 programmatic
state groups, leaving 513 untouched. This experiment hash-ranks those 513
groups using seed 17 and split ID
`native-text-programmatic-group-holdout-v1`; the first 256 are held out and
the remaining 257 untouched groups stay available for fitting. The fit split
also contains the 511 groups seen by the previous sampler, so no inference is
made that they are new data. All rows from a state group stay together. The
expected split is 6,144 fit rows and 2,048 holdout rows, with exactly 768/256
rows per task. A preflight receipt must verify these counts and disjoint row,
group, and observation IDs before training.

The experiment keeps the previous fixed-feature readout and optimizer schedule
unchanged: fresh seed-17 readout initialization, frozen text state and option
features, 2,048 AdamW updates, four microbatches per update, 0.001 peak LR,
62-update warmup, cosine decay, terminal checkpoint only, and no holdout-based
selection. The only experiment-specific settings are the new source/split ID,
the predeclared group counts, and an output subdirectory on local C:.

## Evaluation contract

Evaluation uses only the 2,048 held TRAIN rows. It reports each of the eight
tasks and their equal-task macro, plus an input ablation that zeroes the text
state while keeping candidate-query features fixed. Candidate counts vary
between four and five, so score arrays are padded only for storage and metric
code must ignore padding via the existing `-1e30` sentinel.

No counterfactual donor-target arm is included. Programmatic questions encode
numeric thresholds, and `filter_min_cost` has sparse compatible question and
option-set donors; applying a different group's answer would not preserve the
original question's target semantics. The original and zero-state arms are
paired by state group. These results diagnose this generated task family only;
they do not establish general text competence or product quality.

The corpus and frozen feature banks are pinned to the already staged C: copy.
Only TRAIN rows are read. Official validation and sealed audit remain closed.
The condition-relevance holdout and prior sampler are not retuned. Any failed
attempt is recorded separately; no partial run is resumed as an exact training
resume. All outputs live below
`C:\\CodexArtifacts\\pretrained-reuse-r2-local\\native-text-state-group-holdout\\runs\\programmatic-v1`.

## Attempt 01: fixed terminal result

The C:-local preflight and fixed seed-17 run passed. It completed 2,048
optimizer updates; the terminal checkpoint was reloaded and reproduced every
original holdout logit exactly. The first CLI invocation supplied the preflight
hash in uppercase; hash verification rejected it before creating an output
directory or starting training. The successful invocation used the canonical
lowercase digest. No partial training state was resumed.

| Evaluation arm | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original input | 22.71% | 1.44218 | 0.76258 | 0.03429 macro | 25.08% |
| Text state zeroed | 22.66% | 1.44196 | 0.76248 | 0.03092 macro | 24.86% |

The task-macro uniform-choice reference is 23.75% (six four-option tasks and
two five-option tasks). The original arm is 1.04 percentage points below that
reference; a 5,000-replicate seed-17 state-group bootstrap gives a 95% interval
of [−3.00, +0.96] pp for model accuracy minus uniform-choice accuracy. Thus
the result is near chance and does not establish performance above chance.

| Task | Rows | Uniform reference | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Numeric count | 256 | 20% | 18.75% | 1.60295 | 0.79728 | 0.02346 |
| Cost maximum | 256 | 25% | 18.75% | 1.38886 | 0.75129 | 0.07175 |
| Cost minimum | 256 | 25% | 27.34% | 1.38846 | 0.75103 | 0.02082 |
| Risk maximum | 256 | 25% | 22.27% | 1.38968 | 0.75167 | 0.03517 |
| Risk minimum | 256 | 25% | 23.44% | 1.38412 | 0.74897 | 0.02733 |
| Time maximum | 256 | 25% | 23.83% | 1.38789 | 0.75079 | 0.01950 |
| Time minimum | 256 | 25% | 28.52% | 1.38271 | 0.74820 | 0.02481 |
| Minimum-cost filter | 256 | 20% | 18.75% | 1.61273 | 0.80146 | 0.05148 |

Zeroing the state changed the selected option on 140/2,048 rows (6.84%), so
the state can perturb logits. However, the paired group-bootstrap differences
(zeroed minus original) were Accuracy −0.05 pp [−0.93, +0.78], NLL −0.00021
[−0.00060, +0.00017], and Brier −0.00011 [−0.00029, +0.00007]. The intervals
include zero and show no measurable beneficial state contribution. The
minimum task Accuracy was 18.75%. Training's first/last rolling 128-update
windows were 1.44777/1.44203 CE and 23.05%/23.63% Accuracy; those changing
training windows are not a generalization-gap estimate.

### Execution and artifacts

- Source commit: `985c545c4f4a4b596abcc0a2c85e4833899112df`.
- Config SHA-256: `60e13a82df7b8979ca92684befbf3f6e1d1c95dd625ce855f770afac5d08045b`.
- Preflight SHA-256: `1635dc8d03097670a17650369b94215d38a8c47d5f111701e44a860600860ccd`.
- Split SHA-256: `2a595eee6f442560c83a2f18a22011c68125169058bf70251538dc08d4fde6b2`.
- Run-result SHA-256: `da12b65c4e12b3bcfc10ed0693786158dde95669b97c98ae443c1f69f584cea3`.
- Terminal checkpoint SHA-256: `f87d502d6b127cf529cb9094052f549a8d3c5957f36b772cab8c363f7bf5d533`.
- Holdout-metrics SHA-256: `879df0151d93ad82f37a27f28ef1bd5b66fc94d40b7bca4a6c4c2e088d99f1c4`.
- Saved prediction-array SHA-256: `106fd98eea7a18a058ce140c92db39f16924d325f0c0348b4057ef7a06ba6787`.
- Training-history SHA-256: `ab0453b08b5c47ce3422a3a0496ff86d035c911159e30058d5ee7cd92017fc18`.
- Sampling-record SHA-256: `5a03e46fd1b85a519b1217d3b6a1568784f53147c564f2e72a63ce7f61517514`.
- Start/end: `2026-11-21T06:10:18.725147Z` / `2026-11-21T06:11:25.232159Z`; elapsed 66.51 s, including 46.97 s training and 19.51 s evaluation.
- Local RTX 3080 Laptop; peak CUDA allocated/reserved 208.4/218.0 MiB; 889,345 trainable and 196,254,874 counted prequantization parameters; cloud cost $0.
- Run directory: `C:\\CodexArtifacts\\pretrained-reuse-r2-local\\native-text-state-group-holdout\\runs\\programmatic-v1\\attempt-01`.
- `official_validation_loaded=false`, `sealed_audit_loaded=false`; all fitting and evaluation remained within TRAIN.

This generated source is a weak proxy for broad text reasoning. Its frozen
readout did not beat the uniform-choice reference, and the state-ablation
contrast did not show useful condition binding. Do not spend more capacity or
seeds on this holdout. The next development should diagnose how the frozen
language representation and small candidate readout fail on compositional
state comparison, then test a task with externally grounded multimodal
evidence under a separate TRAIN-only protocol. This result does not qualify
or reject the overall pretrained-reuse objective.
