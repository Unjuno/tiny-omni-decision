# GSM8K MiniLM Decision Probe — Invalidated by Option-Position Leakage

**Disposition: INVALID for model-quality claims. Do not promote, compare, or
select a model from these validation metrics.** The run completed technically,
but a post-run audit found a severe deterministic answer-position leak in the
derived four-choice benchmark. The official GSM8K test split was not accessed.

## What was run

One seed-17 GPU run trained a question-option cross-encoder from the pinned
pretrained `cross-encoder/ms-marco-MiniLM-L6-v2` checkpoint. It did not warm
start from another adapter or checkpoint. Only the final BERT layer and
classifier were trainable (1,774,849 of 22,713,601 parameters). Training used
6,726 examples for three epochs, 20,178 question exposures, and 2,523 updates.
The selected checkpoint was epoch 3 / step 2,523. Best-checkpoint reload and
the frozen-parameter hash check passed.

The base was pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`; the run recorded 90,870,598
weight bytes and SHA-256
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
The data source was `openai/grade-school-math` at revision
`3101c7d5072418e28b9008a6636bde82a006892c`. The source training file SHA-256
was `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`.
The derived split hashes were:

- Train: `611766b764ce7040a3d5f0efa2ee85588e337c440af0dc680c203e52d6736e00`
- Validation: `5487eb941816d1e8c827a4f35189a1638baaf668fa183bc1d13378bb6c1140b5`
- Split manifest: `f9b271746b07287d75fc51cbc778ef8454262ea8359e36e8a58dd4e52e670a87`

The artifacts remain outside Git at:

`C:\CodexArtifacts\pretrained-reuse-r2-local\gsm8k-bert-decision-attempt-01`

The best trainable-weight file SHA-256 is
`8682d7f90eb45c13441c576cebabba340ea453f60abae2d47e99969b56b1e790`; the
final trainable-weight file SHA-256 is
`1712f393bae39e0d0a793f2568dd715231effef9132c9b7141c24237ec5dfe7a`.

The run took 104.08 seconds on an NVIDIA GeForce RTX 3080 Laptop GPU and
reported 372,835,840 peak allocated PyTorch VRAM bytes. Python was 3.11.9,
PyTorch 2.6.0+cu124, Transformers 5.6.2, CUDA runtime 12.4. The recorded
configuration and predictions are retained unchanged.

## Integrity failure

The builder selected distractors with a deterministic SHA-256 ranking and
then used the same hash key to sort the correct answer together with those
distractors. This coupled distractor selection and option ordering: the
selected distractors almost always sorted before the correct answer.

The saved validation split confirms target positions `[26, 24, 28, 669]` for
option indices 0–3. Thus 669 of 747 answers (89.56%) were in the final
position. A trivial “always choose the last option” rule therefore scores
89.56%, making the model’s observed result uninterpretable as reasoning
quality. The same construction affected training and validation.

For transparency only, the recorded validation metrics are:

| Evaluation | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Fresh pretrained baseline | 0.2704 | 1.3825 | 0.7485 | 0.0116 |
| Selected epoch 3 / step 2,523 | 0.4913 | 1.1831 | 0.6384 | 0.0510 |
| Always choose option index 3 | 0.8956 | Not applicable | Not applicable | Not applicable |

These values are **invalid benchmark results**, not evidence for or against
BERT, the training recipe, or general decision ability. The training run and
its checkpoint are retained as an audit artifact, not a candidate. Its
bootstrap interval and question-swap control do not repair the target-position
leak and should not be used to make quality claims. Do not rerun training or
search for a better seed on this exposed split. The official test split was
never read.

## Repair and verification

The builder now uses separate domain-prefixed hashes for distractor selection,
distractor ordering, and target-position assignment. All questions in a
near-duplicate group receive the same target position; positions are assigned
by a deterministic hash rank modulo four. The builder reports per-split target
position counts and rejects gross imbalance (outside 15–35% for any position
when a split has at least 100 examples).

A synthetic regression test creates 120 unique questions and verifies exactly
30 targets at each of the four positions. Focused checks on this code passed:

```text
python -m pytest tests/test_build_gsm8k_choice_split.py -q
7 passed
python -m ruff check scripts/build_gsm8k_choice_split.py tests/test_build_gsm8k_choice_split.py
All checks passed!
```

The corrected builder was **not** used to regenerate or evaluate the already
exposed GSM8K validation records. Its synthetic test is a regression check,
not evidence that a new benchmark is independent. Any future quality study
needs a newly admitted, independently grouped task/split and a frozen option
construction protocol before model fitting.

## Run artifact references

- Run metadata: `effective-config.json`
- Epoch history and recorded validation metrics: `learning-history.json`
- Full run summary: `result.json`
- Saved baseline, per-epoch, selected, and question-swapped predictions are in
  the same external attempt directory.
- The source data audit found no exact normalized question duplicates and one
  approximate near-duplicate pair, which was grouped. Writer identity was not
  available. These limitations are separate from, and do not mitigate, the
  option-position leak.
