# Matched conditional contrast supervision: measured result

## Scope and decision

Two fixed seed 17 fits completed 512 updates each. This is a structured-state
conditional-language experiment, with **zero native media assets**; it does not
qualify text/image/audio/video 95–98% or a complete <=200M product.
Both selected steps: `{'ce': 0, 'contrast': 0}`. Product adoption: **no**.
Predefined combined mechanism gate passed: **False**.
Interpretation and remaining uncertainties are discussed below.

## Frozen design and initialization

[Prospective protocol](R2_CONDITION_CONTRAST_PROTOCOL_20261015.md) and
[configuration](../../configs/pretrained_reuse/condition_contrast_v1.json).
Experiment source commit: `f159d2d7141e056b16b6e617074d31a807649a29`.
Base revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
MiniLM CrossEncoder: 22,713,601 total / 10,647,169 trainable parameters.
Both start from the same coverage-parent delta, with a new AdamW optimizer.
The earlier retained KL512 is a comparison model, not the initialization.
All 6 encoder blocks and classifier train; embeddings/pooler stay frozen.
FP32/SDPA, dropout off, constant LR 2e-5, weight decay 0.01, clip 1.
CE objective `.5*bindingCE + .5*(replayCE + parentKL)`; contrast adds
`.5*.25*contrastCE`, using the same forward inputs/order and budget.
Best/final weights remain separate; no optimizer/RNG resume state was saved.
All experiment code/config/protocol pins remained unchanged during execution.

## Data and accounting

Excluded 6,684 previously used binding states. New state partitions: 512 TRAIN,
128 selection, 256 post-selection; state overlap0. New TRAIN has 2,048 questions
across 8 syntax families; selection has 1,024 questions using 2 different families.
Post-selection has 7 conditions × 1,024 questions on the same 256 states.
Novelty is unused combinations of the existing four attribute vocabularies and
fixed held-out syntax families, not unseen natural concepts or unrestricted language.
These post-selection comparisons are development evidence, not a sealed audit.
No sealed-audit data were loaded.

Each complete fit consumes 4,096 unique questions: 2,048 binding, 1,024 original,
1,024 curriculum. Source-qualified unique states: 512/962/128 respectively.
There are no repeated TRAIN questions; multiple questions share their state.
Across the two matched fits there are 8,192 question presentations but 4,096 distinct
TRAIN questions, because both arms intentionally consume the same examples.
Selected-checkpoint consumed questions: `{'ce': 0, 'contrast': 0}`.
Step 0 means experiment-local fallback, not newly trained weights or promotion.

## Learning curves

Fixed TRAIN is the first 256 binding questions; its metrics are evaluated in a
fixed inference pass. They are not rolling minibatch metrics.

| Arm | Step | Consumed | Fixed TRAIN Acc | TRAIN NLL | Selection Acc | Selection NLL | Brier | ECE | Legacy binding Acc | Parent tasks retained | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ce | 0 | 0 | 25.39% | 3.01822 | 32.32% | 1.84086 | 0.84799 | 0.26280 | 31.74% | True | True |
| ce | 128 | 1024 | 87.50% | 0.33398 | 99.12% | 0.06726 | 0.02203 | 0.04675 | 73.54% | True | False |
| ce | 256 | 2048 | 98.44% | 0.04690 | 100.00% | 0.01404 | 0.00261 | 0.01318 | 85.74% | False | False |
| ce | 384 | 3072 | 100.00% | 0.00360 | 100.00% | 0.00532 | 0.00058 | 0.00513 | 80.76% | True | False |
| ce | 512 | 4096 | 100.00% | 0.00212 | 100.00% | 0.00759 | 0.00127 | 0.00714 | 84.86% | False | False |
| contrast | 0 | 0 | 25.39% | 3.01822 | 32.32% | 1.84086 | 0.84799 | 0.26280 | 31.74% | True | True |
| contrast | 128 | 1024 | 89.84% | 0.28137 | 99.41% | 0.05532 | 0.01830 | 0.04030 | 75.20% | True | False |
| contrast | 256 | 2048 | 100.00% | 0.01893 | 100.00% | 0.01270 | 0.00190 | 0.01210 | 79.69% | False | False |
| contrast | 384 | 3072 | 100.00% | 0.00277 | 100.00% | 0.00740 | 0.00104 | 0.00707 | 83.69% | True | False |
| contrast | 512 | 4096 | 100.00% | 0.00224 | 99.90% | 0.01177 | 0.00329 | 0.01033 | 82.52% | False | False |

The legacy binding floor is 99%, whereas initialization scores 31.74% there.
Failure of that floor is failure to reproduce the retained KL512 capability;
it is not proof of forgetting relative to this run's starting weights.
Parent original/curriculum retention is separately compared against step 0.

## Matched post-selection transfer

All models see the identical 40-option problems. “Known” means this new TRAIN
grammar, not the earlier retained model's training grammar. Identical selected
weights are deduplicated using canonical tensor content, not archive file bytes.

| Model | Condition | Acc | NLL | Brier | ECE | Minimum field Acc | Pair both correct | All4 correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | known | 100.00% | 0.00200 | 0.00011 | 0.00196 | 100.00% | 100.00% | 100.00% |
| control | focus | 31.15% | 2.03691 | 1.00262 | 0.46081 | 0.00% | 4.62% | 0.00% |
| control | complement | 99.80% | 0.05051 | 0.01092 | 0.04369 | 99.22% | 99.61% | 99.22% |
| control | focus_descriptions | 23.14% | 2.72264 | 1.15571 | 0.49294 | 0.00% | 0.59% | 0.00% |
| control | complement_descriptions | 97.27% | 0.25767 | 0.10264 | 0.16844 | 91.80% | 94.53% | 89.06% |
| control | reverse_focus_descriptions | 23.14% | 2.72264 | 1.15571 | 0.49294 | 0.00% | 0.59% | 0.00% |
| control | no_question | 25.00% | 2.45060 | 1.07792 | 0.43842 | 3.52% | 0.00% | 0.00% |
| candidate | known | 100.00% | 0.00265 | 0.00039 | 0.00248 | 100.00% | 100.00% | 100.00% |
| candidate | focus | 26.66% | 2.41699 | 1.15214 | 0.52156 | 0.00% | 1.82% | 0.00% |
| candidate | complement | 100.00% | 0.03745 | 0.00605 | 0.03502 | 100.00% | 100.00% | 100.00% |
| candidate | focus_descriptions | 17.38% | 3.04445 | 1.20794 | 0.52626 | 0.00% | 0.00% | 0.00% |
| candidate | complement_descriptions | 96.39% | 0.28505 | 0.11742 | 0.16800 | 87.50% | 92.77% | 85.55% |
| candidate | reverse_focus_descriptions | 17.38% | 3.04445 | 1.20794 | 0.52626 | 0.00% | 0.00% | 0.00% |
| candidate | no_question | 25.00% | 3.83293 | 1.28359 | 0.60315 | 0.00% | 0.00% | 0.00% |
| retained_reference | known | 35.16% | 2.22924 | 0.85629 | 0.33506 | 25.00% | 25.13% | 25.00% |
| retained_reference | focus | 40.62% | 1.83757 | 0.74603 | 0.09254 | 3.12% | 12.37% | 0.78% |
| retained_reference | complement | 8.50% | 2.83496 | 1.12603 | 0.41973 | 0.00% | 0.00% | 0.00% |
| retained_reference | focus_descriptions | 30.57% | 1.96490 | 0.80455 | 0.16081 | 0.39% | 3.91% | 0.00% |
| retained_reference | complement_descriptions | 1.37% | 2.57719 | 1.00195 | 0.31973 | 0.39% | 0.00% | 0.00% |
| retained_reference | reverse_focus_descriptions | 30.57% | 1.96490 | 0.80455 | 0.16081 | 0.39% | 3.91% | 0.00% |
| retained_reference | no_question | 25.00% | 3.20643 | 1.33143 | 0.64639 | 0.00% | 0.00% | 0.00% |
| control_selected | known | 25.68% | 3.01496 | 1.15981 | 0.49042 | 1.17% | 3.39% | 0.00% |
| control_selected | focus | 27.25% | 2.34852 | 0.98814 | 0.34531 | 1.95% | 1.76% | 0.00% |
| control_selected | complement | 15.04% | 2.30454 | 0.97041 | 0.28252 | 3.52% | 0.20% | 0.00% |
| control_selected | focus_descriptions | 27.25% | 2.00977 | 0.85412 | 0.14465 | 6.25% | 1.82% | 0.00% |
| control_selected | complement_descriptions | 12.99% | 2.34563 | 1.00038 | 0.31596 | 1.95% | 0.00% | 0.00% |
| control_selected | reverse_focus_descriptions | 27.25% | 2.00977 | 0.85412 | 0.14465 | 6.25% | 1.82% | 0.00% |
| control_selected | no_question | 25.00% | 2.43602 | 1.11108 | 0.47692 | 0.78% | 0.00% | 0.00% |

Full per-field metrics/counts, mean confidence, controls and raw-result hashes
are in the adjacent JSON. Primary contrast is **contrast512 minus CE512**.

## Paired uncertainty

10,000 paired whole-state bootstrap draws, seed 17, percentile 95% intervals
with linear quantile interpolation. 256 states are resampled, not 1,024 IID questions.
Intervals condition on these trained weights and these synthetic states; they do
not cover training-seed variability or establish blind product generalization.

| Condition | Acc difference pp | 95% interval pp | NLL difference | 95% interval |
| --- | --- | --- | --- | --- |
| known | +0.000 | [+0.000, +0.000] | +0.00064 | [-0.00010, +0.00156] |
| focus | -4.492 | [-6.055, -3.027] | +0.38008 | [+0.36111, +0.39943] |
| complement | +0.195 | [+0.000, +0.488] | -0.01306 | [-0.01536, -0.01099] |
| focus_descriptions | -5.762 | [-7.324, -4.297] | +0.32181 | [+0.29492, +0.34962] |
| complement_descriptions | -0.879 | [-1.758, +0.000] | +0.02738 | [+0.02160, +0.03304] |
| reverse_focus_descriptions | -5.762 | [-7.324, -4.297] | +0.32181 | [+0.29492, +0.34962] |
| no_question | +0.000 | [+0.000, +0.000] | +1.38233 | [+1.28841, +1.47699] |

Predefined combined mechanism failures: `final_retention, focus_descriptions:improvement, focus_descriptions:paired_lower, focus_descriptions:nll, focus_descriptions:brier, focus_descriptions:ece, complement_descriptions:improvement, complement_descriptions:paired_lower, complement_descriptions:nll, complement_descriptions:brier`.

Quality failures: `focus_descriptions:minimum_task_accuracy, focus_descriptions:pair_accuracy, focus_descriptions:improvement, focus_descriptions:paired_lower, focus_descriptions:nll, focus_descriptions:brier, focus_descriptions:ece, complement_descriptions:minimum_task_accuracy, complement_descriptions:improvement, complement_descriptions:paired_lower, complement_descriptions:nll, complement_descriptions:brier`.

## Post-hoc error description (not a selection criterion)

This uses saved predictions only and changes no experiment gates. The new focus
sentence is “Your output should concern {field}; values for {others} do not answer
this request.” The new complement sentence permits the one property outside a
forbidden list. Results establish sensitivity to these frozen syntax families,
not all unseen English, all negation, or a fundamental model-capacity limit.

| Model | Condition | Questions | Correct | Wrong field | Right field / wrong value | Wrong field / value matches state |
| --- | --- | --- | --- | --- | --- | --- |
| control | focus_descriptions | 1024 | 237 | 787 | 0 | 787 |
| control | complement_descriptions | 1024 | 996 | 28 | 0 | 28 |
| candidate | focus_descriptions | 1024 | 178 | 846 | 0 | 846 |
| candidate | complement_descriptions | 1024 | 987 | 37 | 0 | 37 |
| retained_reference | focus_descriptions | 1024 | 313 | 711 | 0 | 711 |
| retained_reference | complement_descriptions | 1024 | 14 | 1010 | 0 | 1010 |
| control_selected | focus_descriptions | 1024 | 279 | 745 | 0 | 745 |
| control_selected | complement_descriptions | 1024 | 133 | 891 | 0 | 891 |

Per-field primary-condition metrics for both fixed 512 endpoints:

| Model | Condition | Field | Count | Acc | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | focus_descriptions | color | 256 | 0.78% | 3.92348 | 1.53460 | 0.73320 |
| control | focus_descriptions | count | 256 | 0.00% | 4.48893 | 1.66229 | 0.79776 |
| control | focus_descriptions | shape | 256 | 64.06% | 0.81844 | 0.44812 | 0.10529 |
| control | focus_descriptions | texture | 256 | 27.73% | 1.65971 | 0.97784 | 0.45426 |
| control | complement_descriptions | color | 256 | 97.27% | 0.29965 | 0.13047 | 0.21202 |
| control | complement_descriptions | count | 256 | 91.80% | 0.51291 | 0.22011 | 0.27016 |
| control | complement_descriptions | shape | 256 | 100.00% | 0.12247 | 0.03620 | 0.10619 |
| control | complement_descriptions | texture | 256 | 100.00% | 0.09565 | 0.02376 | 0.08537 |
| candidate | focus_descriptions | color | 256 | 1.95% | 3.96332 | 1.47916 | 0.68395 |
| candidate | focus_descriptions | count | 256 | 0.00% | 5.23545 | 1.64419 | 0.76263 |
| candidate | focus_descriptions | shape | 256 | 31.64% | 1.49988 | 0.85141 | 0.31711 |
| candidate | focus_descriptions | texture | 256 | 35.94% | 1.47913 | 0.85699 | 0.35156 |
| candidate | complement_descriptions | color | 256 | 98.05% | 0.32157 | 0.13966 | 0.23768 |
| candidate | complement_descriptions | count | 256 | 87.50% | 0.67814 | 0.30238 | 0.30482 |
| candidate | complement_descriptions | shape | 256 | 100.00% | 0.06057 | 0.00992 | 0.05650 |
| candidate | complement_descriptions | texture | 256 | 100.00% | 0.07993 | 0.01772 | 0.07301 |

## Verification and resources

Each selected fresh-process replay matches 15 files exactly.
Full post-selection fresh replay matches 86 files exactly.
The publisher recomputes metrics, retention, selection, probability records,
controls and bootstrap results from saved scores. It requires exact corpus/code
provenance and links both transfer manifests to the current two fit manifests.

RTX 3080 Laptop 16GB; environment: `{"python": "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]", "torch": "2.6.0+cu124", "numpy": "2.4.6", "cuda": "12.4", "gpu": "NVIDIA GeForce RTX 3080 Laptop GPU", "packages": {"transformers": "5.6.2", "tokenizers": "0.22.2", "safetensors": "0.8.0", "psutil": "7.2.2"}, "attention": "sdpa", "dtype": "torch.float32", "threads": 4}`.

| Process | Wall seconds | Peak CUDA allocated bytes | Peak host bytes | Start UTC | Finish UTC |
| --- | --- | --- | --- | --- | --- |
| condition-contrast-ce-fit-01 | 596.86 | 2257383424 | 1756020736 | 2026-10-15T07:57:20.444085+00:00 | 2026-10-15T08:07:17.310473+00:00 |
| condition-contrast-ce-replay-01 | 92.68 | 338388992 | 1310322688 | 2026-10-15T08:19:15.059127+00:00 | 2026-10-15T08:20:47.741743+00:00 |
| condition-contrast-fit-01 | 611.49 | 2257383424 | 1761501184 | 2026-10-15T08:08:33.248306+00:00 | 2026-10-15T08:18:44.740725+00:00 |
| condition-contrast-replay-01 | 93.57 | 338388992 | 1310408704 | 2026-10-15T08:20:48.993951+00:00 | 2026-10-15T08:22:22.568264+00:00 |
| condition-contrast-transfer-01 | 551.64 | 356538368 | 1300709376 | 2026-10-15T08:23:01.948416+00:00 | 2026-10-15T08:32:13.594073+00:00 |
| condition-contrast-transfer-replay-01 | 557.17 | 356538368 | 1300615168 | 2026-10-15T08:32:57.187974+00:00 | 2026-10-15T08:42:14.361108+00:00 |
| condition-contrast-smoke-01 | 33.66 | 1947896832 | 1723486208 | 2026-10-15T07:56:27.079101+00:00 | 2026-10-15T07:57:00.737388+00:00 |

Summed measured process-body wall time: 2,537.07 seconds (42.28 minutes).
This includes inference/evaluation/loading/I/O and is not GPU-kernel time.
Cloud cost 0; local electricity cost was not measured. Training cache logits were
reused with full row/weight hashes, so new teacher-forward time was 0.
Smoke weights were discarded. CPU verification overlapped portions of GPU processes;
wall-time differences between arms are not an isolated loss-overhead benchmark.

Legacy long-text truncation remains the prior only-second 512-token policy.
Each fit reports 1,624 truncated candidate presentations; new binding/transfer
truncation is forbidden. This preserved policy limits long-text conclusions.

## Artifacts and reproduction

Artifact root: `C:/CodexArtifacts/pretrained-reuse-r2`. No weights or media are added to Git.
Publication SHA256: `12b5c013c9bd03f3cc06eba535431461e7e53f1eb815a865c946b870406db1fc`.
Corpus manifest SHA256: `78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9`.

| Corpus artifact | SHA256 |
| --- | --- |
| accounting.json | c2fde5591129a939d902a1337729bf4fdbff5a816b9fedbe24caf2d3ea5a2af8 |
| complement.jsonl | 09fb616922c8a24d2f89b3d1853eafffb2059e9a8e050b5ab672800d90a9c4b6 |
| complement_descriptions.jsonl | c6c3542b220d18b4d1f77cbc06a434aa464017a8cbe2ab6c7a86ee29e883aa2c |
| focus.jsonl | 48ab099ad113122bd141100e9c67ffa6f83e4d51090d0a63732894f55d88d49a |
| focus_descriptions.jsonl | aeebc813e7c8ea04f4be3c6410c34174b96d0720a85841aa016ddcd56eef32de |
| known.jsonl | c8050cba56c035a95166b49978f623daf5d0c852888c5915e5e8a6392d5a8ff7 |
| no_question.jsonl | e5f0ecae1caaf584592d97d7d38f5edad1bb3f4d797f032f5b950b2abdb33bc6 |
| replay.jsonl | a6d363fe4349432ecaf2a29465ee166e7f53ddf577d930884afaa0327df0814c |
| reverse_focus_descriptions.jsonl | 21a7811a6ca6adabdd021e2893f697c9676100e7fa36a5474532c43c06028aea |
| sample-order.json | bf6b497fb66b990bbe7955eecc8fa6398374363a8fe2ba1185ac75ab69e3fdd9 |
| state-partitions.json | 89823041cafc3e678c3f0d6c1e92b05da0c437ad5da2873ce5ee627d9dd8dd48 |
| train.jsonl | 7f57e5ae3ea67487518b2121d0120d37dc62f8816d6c01990c2b3f2909fd7ea3 |
| validation.jsonl | d5287c01ad98a4ed857d038bdad7f810cc1ee86399027221c37fcc1c269af42e |

| Publication verifier source | SHA256 |
| --- | --- |
| C:\CodexWorktrees\pretrained-reuse-omni\scripts\publish_condition_contrast.py | 254220fa9b6f7bbc53d57cf35b48b1febcf6473570fa4722b55cf7a3adb2692b |
| C:\CodexWorktrees\pretrained-reuse-omni\tests\test_condition_contrast_publication.py | 5c579e0d5dec06fbccf9791226038c322e2e6a6d0629bcb3f2e82e7bf7d51b38 |

| Run manifest | SHA256 |
| --- | --- |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-ce-fit-01\result.json | 3864e669dfedce7545f84f60a35121d911923dbf18a002d57e3900a328fc59de |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-ce-replay-01\result.json | 149dd2858cffae1a09a3f44edd4dec4f1e18a58fa83aa5669bf5709ac92ccd19 |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-fit-01\result.json | 13e4fc327fec2d02ec07eca2c4ea47e2ec9feb5c153e06e4cee64d63b764bf5e |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-replay-01\result.json | 1dbefdf7f8dd0f125320691d91d51079af1590074b54214bd6d9e952a4f734fa |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-transfer-01\result.json | 77374d662cf68a3df95eae01d72375b107d1b8974811234c5af0accad00e5233 |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-transfer-replay-01\result.json | 57477ec51094ed0fbdbfaab7c0a1b925324b3e4a3eea30a8e2d987d7729781e8 |
| C:\CodexArtifacts\pretrained-reuse-r2\condition-contrast-smoke-01\result.json | 12834d7e581fbd561be0e36e8f90c6011cdd80629d64697a9c5108beddd1f27d |

Exact process argv are preserved in each run manifest. Reproduction commands
below used Python 3.11.9; all output directories must be new.

### condition-contrast-ce-fit-01

```text
python -m scripts.train_condition_contrast --arm ce --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-ce-fit-01
```

### condition-contrast-ce-replay-01

```text
python -m scripts.train_condition_contrast --arm ce --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-ce-fit-01 --replay-sha256 3864e669dfedce7545f84f60a35121d911923dbf18a002d57e3900a328fc59de --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-ce-replay-01
```

### condition-contrast-fit-01

```text
python -m scripts.train_condition_contrast --arm contrast --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-fit-01
```

### condition-contrast-replay-01

```text
python -m scripts.train_condition_contrast --arm contrast --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-fit-01 --replay-sha256 13e4fc327fec2d02ec07eca2c4ea47e2ec9feb5c153e06e4cee64d63b764bf5e --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-replay-01
```

### condition-contrast-transfer-01

```text
python -m scripts.evaluate_condition_contrast --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --ce-fit C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-ce-fit-01 --ce-sha256 3864e669dfedce7545f84f60a35121d911923dbf18a002d57e3900a328fc59de --contrast-fit C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-fit-01 --contrast-sha256 13e4fc327fec2d02ec07eca2c4ea47e2ec9feb5c153e06e4cee64d63b764bf5e --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-transfer-01
```

### condition-contrast-transfer-replay-01

```text
python -m scripts.evaluate_condition_contrast --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --ce-fit C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-ce-fit-01 --ce-sha256 3864e669dfedce7545f84f60a35121d911923dbf18a002d57e3900a328fc59de --contrast-fit C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-fit-01 --contrast-sha256 13e4fc327fec2d02ec07eca2c4ea47e2ec9feb5c153e06e4cee64d63b764bf5e --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-transfer-01 --replay-sha256 77374d662cf68a3df95eae01d72375b107d1b8974811234c5af0accad00e5233 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-transfer-replay-01
```

### condition-contrast-smoke-01

```text
python -m scripts.train_condition_contrast --arm contrast --smoke --corpus C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-corpus-01 --corpus-sha256 78b803f6e557d73b1c2dcff7ce60a46ca2ca62d9034941a6b4d60754a5685ff9 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-smoke-01
```

| Run | Weight artifact | Bytes | SHA256 |
| --- | --- | --- | --- |
| condition-contrast-ce-fit-01 | best-delta.pt | 42620884 | b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5 |
| condition-contrast-ce-fit-01 | final-delta.pt | 42620986 | 31960b6ec5db4df3d61b66bd34a6d2346c873cbee93faf69b0690f19d12b14e7 |
| condition-contrast-fit-01 | best-delta.pt | 42620884 | b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5 |
| condition-contrast-fit-01 | final-delta.pt | 42620986 | 1bf937f869993300fadffd2eb11f7ada8d908552fc43cb775f75646faaa05723 |

## Interpretation and next action

**Execution PASS. Primary contrast mechanism: NOT SUPPORTED WITHIN THIS DESIGN
AND BUDGET. Neither newly trained endpoint is adopted.** Both primary conditions
with new candidate descriptions worsen: focus Accuracy changes by -5.762 pp
(95% paired interval [-7.324, -4.297]), and complement by -0.879 pp
([-1.758, 0.000]). NLL and Brier worsen for both. The secondary complement
condition with original descriptions improves NLL and gains two correct answers;
its Accuracy interval includes zero. This limited secondary gain does not satisfy
the predefined requirement to improve both primary conditions and retain old tasks.

The ordinary CE endpoint already reaches 97.27% on the new complement-description
family, while failing the focus-description family at 23.14%. Thus the result
does not support an assertion that exclusion is inherently impossible for this
model. It also does not establish general conditional understanding. The earlier
KL512 reference is stronger on focus-description (30.57%) but much weaker on
complement-description (1.37%). Training data, grammar, replay composition and
grouped batches differ from that historical run. All three fits share the same
coverage-parent initialization; only the two new matched arms isolate the auxiliary loss.
All research checkpoints are retained; none qualifies across the required tasks.

The post-hoc error description points to conditional field selection in these
syntaxes: all 787 CE and 846 contrast focus-description errors choose another
field's correct state value. It does not isolate language-encoder capacity from
training coverage, learned scoring biases or representation. These are oracle
state inputs, so the failure cannot be attributed to image/audio/video perception
in this experiment. The no-question control is deliberately ambiguous across
four attributes; its 25% is not a requirement that native product tasks use text.
Candidate-order identity agreement is 100%; maximum probability change is
4.45e-16. Order stability is established for this scorer, not semantic correctness.

There is a late overfitting warning in both curves: from 384 to 512, fixed TRAIN
NLL continues falling while selection NLL rises (CE .005319 to .007586;
contrast .007397 to .011772). The final fixed-TRAIN/selection Accuracy gaps are
0 and 0.098 pp, which conceal the much larger syntax-transfer failure. This is
neither a clean capacity ceiling nor justification for an unplanned longer run.

Parent-task retention also fails independently of the legacy 99% floor. At 512,
CE drops the original `noul` task by 1.520 pp and curriculum `count` by 4.688 pp.
Contrast drops these by 1.013 and 3.516 pp and the textual curriculum
`temporal_order` task by 1.172 pp. These are text retention tasks, not native
video-quality measurements. Every trained checkpoint fails the combined selector;
the untrained experiment fallback does not replace the historical reference.

Close this fixed loss comparison. Do not sweep coefficients, templates, budgets
or seeds on these observed transfer states. Before another semantic fit, compare
the suitability of pretrained conditional-language representations and their
whole-product parameter allocation, using a separately fixed contract. Keep
numeric and legacy decisions in that contract, and test both positive focus and
exclusion with irrelevant mentions. A larger language component is a hypothesis
to measure within the 200M total, not an automatic solution. Retain the native
input/JSON pipeline; once conditional quality is established, connect the same
questions to native features and measure the perception gap, temporal reasoning
and genuinely complementary multimodal inputs. No quantization or model promotion.

## CPU/CI checks

Ruff passed. CPU suite: 1,636 passed / 1 skipped (23.72 seconds; final log `condition-contrast-pytest-04.log`); all 25 manifest
commands passed. Targeted publication tests pass after adding verifier source pins.
The first manifest harness invocation flattened a PowerShell argument array and
failed before model validation; corrected invocation02 passed all 25 commands.
Provenance regressions were added red-to-green and reviewed before publication.
CI run 1757 passed on fe4c954. Final result-commit CI evidence is recorded
separately at `C:/CodexArtifacts/pretrained-reuse-r2/condition-contrast-results-ci-01.json`
after the push, avoiding a self-referential report commit hash.
