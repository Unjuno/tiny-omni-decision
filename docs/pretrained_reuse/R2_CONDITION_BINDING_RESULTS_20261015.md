# Condition binding: learned the new task, failed retention

One fixed local seed17 continuation completed all512 optimizer updates. New
binding validation reached100%, but **every trained checkpoint failed the
predefined retention gates**. The selected checkpoint is therefore step0, the
unchanged parent. The final trained weights are preserved separately and are
not promoted. The <=200M native four-modality95–98% product goal remains unmet.

[Prospective protocol](R2_CONDITION_BINDING_PROTOCOL_20261015.md) ·
[Machine-readable evidence](R2_CONDITION_BINDING_RESULTS_20261015.json) ·
[Strategy](STRATEGY_20261015.md).

## What was actually compared

Control: retained22,713,601parameter text coverage parent. Candidate: the same
architecture/initialization plus512 fixed AdamW updates,4binding+4original replay
questions per update, equal source losses, LR2e-5constant, seed17, FP32SDPA,
dropout disabled. All six encoder blocks and classifier train (10,647,169params);
embeddings/pooler stay frozen. Optimizer reset; this is not exact resume.

Binding TRAIN contains2,048questions over512distinct annotation-derived states.
Original replay contains2,048questions over1,815distinct states, exactly one
presentation per question during this fit. Replay counts: choice814/noul824/
score410. The full fit consumes4,096unique question IDs, zero repeated IDs,
2,327source-qualified states. Selected step0 consumes **zero additional fit
questions**; the parent's historical training is not zero and is documented in
the earlier coverage experiment. No actual media are consumed here.

The selector has1,024questions on256new combinations excluded against all5,636
distinct states in the full8,424image upstream TRAIN pool, plus308states observed
by the preceding oracle/predicted diagnostic. Transfer has256different new
combinations. The same40typed choices remain available for every question.
These are semantic combinations within the same taxonomy, not new real-image
scenes or previously unseen concepts. Exclusion does not claim an audit against
every historical media validation split; no such media were opened.

Retention evaluation keeps all2,540original and2,048curriculum validation
questions in the parent's order. These are observed regression benchmarks.
The new fit replays original TRAIN, not curriculum TRAIN. A replay-only trained
control is absent, so the observed changes cannot isolate the causal effect of
binding examples from all effects of continuing training.

## Fixed learning curve

Accuracy in percent; NLL uses stable FP64 log-softmax, Brier sums across classes,
ECE uses15bins. Every checkpoint includes all examples and all fields/tasks.

| Step | Consumed questions | Binding Acc | Binding NLL | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
|0|0|31.74|2.189042|63.58|.835591|84.08|.375370|yes, selected|
|128|1,024|100.00|.001238|63.15|.892758|83.30|.448659|no|
|256|2,048|100.00|.001133|64.06|.943513|84.03|.479282|no|
|384|3,072|100.00|.000987|62.99|.942825|84.13|.514190|no|
|512|4,096|100.00|.000800|63.54|.898312|84.23|.437499|no|

The fixed256question TRAIN probe also reaches100% from step128. Its final NLL is
.000051 versus new-binding validation .000800; Accuracy gap0. There is no observed
late decline on this narrow binding validation, but this is not proof of broad
language transfer or absence of interference with other tasks. Every field has
256validation questions and100% final Accuracy; all-four and six-pair correctness
both become100%. The earlier frozen diagnosis's27.05% uses different states and
questions; compare this run's31.74% baseline with its own final100%.

## Why no new checkpoint was accepted

Eligibility requires every old task's Accuracy to stay within1percentage point,
and each old source's NLL/Brier/ECE to increase by no more than.02/.01/.02.
Selection is lowest binding NLL among eligible points, earliest tie. These
thresholds were not changed after results. Step0 is the valid fallback.

| Metric | Original parent | Original step512 | Curriculum parent | Curriculum step512 |
|---|---:|---:|---:|---:|
|Accuracy|.635827|.635433|.840820|.842285|
|NLL|.835591|.898312|.375370|.437499|
|Brier|.470606|.485632|.214324|.228650|
|ECE|.018915|.097916|.016823|.043234|
|Mean confidence|.642509|.731912|.836965|.879768|

Aggregate Accuracy nearly holds, while confidence rises and probability quality
worsens. Final original score Accuracy falls49.91%→48.62% (−1.28pp), and curriculum
extremum89.65%→88.57% (−1.07pp), also violating task gates. All NLL/Brier/ECE source
gates fail at512. Full per-task numbers and violations at all four trained
checkpoints are in the JSON. This is measured interference/calibration drift;
neither universal model incapacity nor severe forgetting of every task follows.

## Transfer, controls and reproducibility

Only the frozen selected checkpoint and parent are evaluated on the four
familiar/novel question × familiar/novel candidate-description conditions.
Because selection retained step0, this comparison evaluates identical weights.
The rejected final checkpoint is **not** scored on transfer. Its transfer quality
therefore remains UNKNOWN; the narrow100% result cannot be called generic
condition understanding. The old oracle/predicted-state probe is regression only.

Paired uncertainty resamples complete four-question states10,000times with
NumPy default_rng seed17, percentile95% intervals and linear quantiles. The
mean remains question-weighted; correlated questions are never independent draws.
Selector intervals carry selection/development bias, and the authored transfer
is one wording family with no multiplicity correction. This is not a blind audit.

The final-versus-parent binding validation gain is68.26pp, paired95% interval
[66.80,69.73]pp; NLL change−2.18824, interval[−2.25242,−2.12172]. This is an
unselected endpoint on development validation. Parent-to-selected differences
are exactlyzero with zero-width intervals: selection retained the same weights.
These degenerate intervals describe identical predictions, not certainty about
population performance.

| Question / candidate wording | Parent and selected Acc | NLL | Brier | ECE | All four correct | Pairs both correct |
|---|---:|---:|---:|---:|---:|---:|
|Familiar / familiar|.309570|2.099754|.931476|.327158|0|.046224|
|Familiar / novel|.550781|1.281476|.588223|.104897|.054688|.277344|
|Novel / familiar|.342773|1.903379|.854915|.292702|0|.066406|
|Novel / novel|.288086|1.900426|.880808|.337102|0|.026042|

Each row is the same256states/1,024questions, not four independent populations.
Question wording and candidate descriptions interact strongly; do not select
the55.08% wording as a new preferred benchmark. These conditions were authored
before fitting and none is used to select a checkpoint or change training.
Question-absent Accuracy25%, all-four0, pairs0. Reversing all40candidates gives
zero mapped-score error and zero answer changes for both models. The old probe
retains27.05% oracle /23.93% predicted-state Accuracy; tiny floating-point NLL
differences from its older run accompany changed batching and are not improvement.

## Actual execution and saved artifacts

Data source commit `7502f4b5e1e9712e384e709e099a40370ba0fa87`.
Training/replay source commit `37ede80c91addd7df659054b7eb878288dafe0f1`.
Base `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`; initialization uses the preserved
coverage-parent dense encoder/classifier delta, not a new pretrained model.
Delta weights contain no optimizer/scheduler/RNG state for exact resume.

Commands from `C:/CodexWorktrees/pretrained-reuse-omni`:

```powershell
python -m scripts.build_condition_binding_corpus --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-corpus-01
python -m scripts.train_condition_binding --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-smoke-01
python -m scripts.train_condition_binding --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-fit-01
python -m scripts.train_condition_binding --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-selected-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-fit-01 --replay-sha256 d58df7c1e6bb4cda0f63b8e92554699f40e53baef65d53037fad583434684cb4
```

These commands were executed once in distinct directories; their runners refuse
existing outputs. The final command performs inference only, not another fit.
Do not recreate or overwrite the frozen data to reproduce analysis.

RTX3080Laptop16GiB, Windows/Python3.11.9, PyTorch2.6.0+cu124/CUDA12.4,
Transformers5.6.2, NumPy2.4.6, tokenizers0.22.2, safetensors0.8.0,
psutil7.2.2. FP32SDPA, TF32off, deterministic kernels,4CPUthreads; no environment
upgrade. The model has22,713,601total /10,647,169trainable parameters.

| Process | Updates | Wall seconds | Peak CUDA allocated bytes | Peak CUDA reserved bytes | Peak host bytes |
|---|---:|---:|---:|---:|---:|
|TRAIN-only smoke|2|8.587|1,720,447,488|1,944,059,904|1,545,179,136|
|Primary fit and evaluation|512|697.404|2,067,150,848|2,596,274,176|1,609,146,368|
|Fresh selected inference replay|0|288.066|306,067,456|417,333,248|1,349,140,480|

Primary began2026-10-15T03:59:10.822427Z and ended04:10:48.226363Z after all
scheduled updates, selection, reload and fixed controls. Optimizer update bodies
total128.990s (0.252s/update); five scheduled evaluations total254.121s. The rest
includes loading, hashing, selected reload and transfer/control inference.
These are process timings, not dedicated GPU-busy seconds or isolated throughput
benchmarks; CPU verification also ran during post-training inference. No cloud
or paid compute. All trainable gradients were finite/nonzero; recorded preclip
norm range.13724–48.09299, clipped to1. Frozen weights and all source pins pass.

Primary makes1,157,926candidate-pair forward presentations including repeated
evaluation; this is **not** the4,096unique training-question count. There are
1,779truncated old-source pair presentations across163unique question IDs under
the preserved parent's512token `only_second` policy. New binding has no
truncation. The tokenizer's initial513>512warning comes from the untruncated
length audit; actual model inputs never exceed512tokens.

Root `C:/CodexArtifacts/pretrained-reuse-r2`:

| Artifact | SHA-256 |
|---|---|
|`condition-binding-corpus-01/result.json`|`957e2ce3ade65c5d2caa26a1cbb0e95a3adeee5cf5b312110f8c2a7e3cfed971`|
|`condition-binding-fit-01/result.json`|`d58df7c1e6bb4cda0f63b8e92554699f40e53baef65d53037fad583434684cb4`|
|`condition-binding-fit-01/effective-config.json`|`959dcd8641e09b72d782bcfe4ca585e13430302fb70dddcc55c066f9f9679c76`|
|`condition-binding-fit-01/best-delta.pt`|`b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5`|
|`condition-binding-fit-01/final-delta.pt`|`3be52d41760588fe39bf11928c7fa4fc2129e8a6e22b16da783eeea9bc448ab9`|
|`condition-binding-selected-replay-01/result.json`|`8900544fb1731e009d3f9c8355973823a32fdee9f5f52b8f7e09a6d31f5e01d6`|
|`condition-binding-publication-01.json` (local publication; tracked JSON has the same content)|`8919a8cfa1ec49924bb7d5f8a23d8361329c8d59893c6b0393d3e91b2f152784`|

Best is byte-identical to the preserved parent delta (42,620,884bytes); final is
a different42,620,986byte artifact. Both, all scores/probabilities, full history,
updates, IDs and corpus hashes remain local. No weights or media enter Git.
Fresh replay ran2026-10-15T04:11:05.046078Z–04:15:53.111166Z. All four selected
validation/fixed-TRAIN raw arrays match the primary saved arrays exactly;
36transfer/regression score/probability/metric payloads and both aggregate
transfer/permutation files match byte-for-byte. Source and frozen-weight hashes
also pass. This is inference reproducibility, not a second training-seed result.
Smoke+primary+fresh process-body wall time totals994.057s (16.57minutes), local
compute only; dedicated GPU-busy time was not separately measured.

The CPU-only publication verifies every primary/replay artifact and source pin,
reconstructs selection/gate failures, and computes state-paired intervals:

```powershell
python -m scripts.summarize_condition_binding --root C:/CodexArtifacts/pretrained-reuse-r2 --primary-sha256 d58df7c1e6bb4cda0f63b8e92554699f40e53baef65d53037fad583434684cb4 --replay-sha256 8900544fb1731e009d3f9c8355973823a32fdee9f5f52b8f7e09a6d31f5e01d6 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-publication-01.json
```

Local verification: Ruff passes; CPU suite1,586passed/1skipped; simulated-no-Torch
suite1,159passed/109skipped; model/dataset manifests and all four catalog
validations pass. Validating REVIEW/DENY metadata does not admit those sources
to training. Read-only code review found no blocking computational/selection
defect; two documentation ambiguities about trained encoder weights and the
smoke were corrected. GitHub Actions checks are associated with the exact
publication commit on Draft PR15; no main merge.

## Strategy consequence

This supports the priority identified by the oracle diagnosis: explicit
condition/field supervision can fix that narrow failure without larger encoders
or more parameters. It also shows why new-task Accuracy alone is insufficient.
The next hypothesis should protect old decisions and their probability scale
during learning, with an unchanged task gate. Do not add steps, tune temperature
on this validation, rescue the model through new transfer wording, or relax the
retention threshold to claim progress.

A bounded next comparison can change only the retention objective, preserving
the parent's option distributions on existing TRAIN replay as well as labels;
it requires its own frozen protocol before execution. This is output-level
retention from the existing parent, not a claim that the parent is a stronger
general teacher or a repeat of whole-embedding self-distillation. Keep the
current failed candidate as the matched control. Any future novel-transfer
claim needs an independently specified condition family; the exposed transfer
sets become regression evidence.

Native text/image/audio/video integration, real temporal/AV reasoning, OOD
quality, full resident size and product calibration remain open. Existing
encoder/Teacher artifacts, sealed audit, main branch and production model remain
untouched; candidate text-encoder deltas are separate new artifacts. One separate
two-update TRAIN smoke preceded the sole full fit. No additional seed, second
full fit, cloud job or quantization was started.
