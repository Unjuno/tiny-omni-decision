# Fixed candidate: independent conditional transfer

[Frozen protocol](R2_BINDING_TRANSFER_PROTOCOL_20261015.md) ·
[Complete machine-readable evidence](R2_BINDING_TRANSFER_RESULTS_20261015.json) ·
[Prior retention comparison](R2_BINDING_RETENTION_KL_RESULTS_20261015.md).

## Outcome and next decision

**Execution and full fresh-process replay PASS; predefined transfer quality FAIL.**
The fixed retained candidate scores 100% on familiar instruction/option wording
with unused state combinations. It scores 49.22% when both question and candidate
descriptions change, and 9.47% when asked to ignore other attributes with new
candidate descriptions. Both predeclared primary transfer arms fail their gates.
No model updates, calibration, prompt search, checkpoint selection or extra seed.

Partial transfer is real: the joint wording change improves 14.55 percentage
points over the parent. Exclusion instructions with new descriptions instead
lose 13.09 points. Preserve both observations. Neither narrow 100% nor better
ECE in an unfavorable arm establishes useful general conditional reasoning.
Do not promote this candidate to the product or change the acceptance rules.

Next priority is diverse conditional supervision: same words with different
requested roles, exclusion/negation, and meaning-preserving candidate descriptions.
Predefine independent training and held-out syntax families. Keep this now-observed
set as a regression, not a new selector or blind audit. Do not add its exact failed
questions to training and describe that as transfer. The evidence does not justify
an encoder/capacity ceiling or another budget/parameter increase on the same data.

## Frozen models, data and controls

Source commit: `a66fbb4f9c105a3b5b1370bc2bee1ded84e1926c`.
Both models have 22,713,601 parameters, all frozen in this comparison. The base is
`cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Exact encoder/classifier key coverage is
required when loading each delta; the remaining base tensors are shared unchanged.

- Parent: `text-coverage-training-01/best-delta.pt`, SHA-256
  `b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5`.
- Candidate: `condition-binding-retention-fit-01/best-delta.pt`, selected step 512,
  SHA-256 `c87ddf2197e276814e4941efb7ad7431d416316eb786ea59b8331925ca16ab02`.

The corpus is prepared before model inference. Exclude the union of **6,172**
state identities from all upstream TRAIN combinations, old diagnostic oracle and
predicted states, and prior binding train/validation/transfer partitions. Select
256 of the remaining 3,828 by the frozen SHA ordering. There is zero identity and
canonical-state overlap with that union. No example is removed or replaced after
scoring. The old transfer partitions are read only to exclude states; old transfer
questions/metrics and sealed audit are not loaded or evaluated.

Each new state has four distinct requested attributes and all 40 typed options.
All eight conditions retain 1,024 aligned targets. There are **256 independent
state groups**, not 8,192 independent scenes; eight arms intentionally reuse the
same questions/states. Individual values are known, combinations are unused.
Each field has ten values, occurring 16–33 times in this cohort; counts are in JSON.
No image, audio, video or sensor is consumed. The state is an annotation-derived
four-fact JSON representation. These are English generated development tasks,
with one new paraphrase and one exclusion instruction per field, not broad
language diversity or a final product audit.

`known` keeps training-style wording; `question` changes only the question;
`options` changes only candidate descriptions; `joint` changes both. `distractor`
mentions all three irrelevant attributes then requests the remaining field;
`distractor_options` also changes descriptions. `no_question` removes the condition;
`reverse_joint` reverses the complete candidate list while retaining identities.
Exact strings are in the committed helper. All arms are reported; none was chosen
as a better prompt. Both primary arms were fixed before execution.

## Full measured comparison

| Condition / model | Accuracy | NLL | Brier | ECE | All four | Pair both |
|---|---:|---:|---:|---:|---:|---:|
| known / parent | 31.35% | 2.101636 | 0.924220 | 0.332875 | 0.00% | 4.43% |
| known / candidate | 100.00% | 0.001727 | 0.000383 | 0.001619 | 100.00% | 100.00% |
| question / parent | 30.47% | 1.798315 | 0.851768 | 0.234305 | 0.00% | 4.23% |
| question / candidate | 62.11% | 0.836920 | 0.465492 | 0.087568 | 7.03% | 34.64% |
| options / parent | 62.21% | 1.193221 | 0.503124 | 0.088078 | 8.59% | 35.55% |
| options / candidate | 88.28% | 0.429618 | 0.135524 | 0.053610 | 53.12% | 76.56% |
| joint / parent | 34.67% | 1.790671 | 0.768121 | 0.134884 | 0.00% | 6.58% |
| joint / candidate | 49.22% | 1.452757 | 0.623361 | 0.140692 | 1.17% | 19.79% |
| distractor / parent | 21.68% | 3.125326 | 1.225503 | 0.524483 | 0.00% | 0.00% |
| distractor / candidate | 13.87% | 2.501955 | 1.008786 | 0.296871 | 0.00% | 0.00% |
| distractor_options / parent | 22.56% | 2.998154 | 1.059654 | 0.345191 | 0.00% | 0.07% |
| distractor_options / candidate | 9.47% | 3.007345 | 0.982775 | 0.196100 | 0.00% | 0.00% |
| no_question / parent | 24.90% | 2.532819 | 1.090858 | 0.457729 | 0.00% | 0.00% |
| no_question / candidate | 25.00% | 2.075317 | 0.880555 | 0.218688 | 0.00% | 0.00% |
| reverse_joint / parent | 34.67% | 1.790671 | 0.768121 | 0.134884 | 0.00% | 6.58% |
| reverse_joint / candidate | 49.22% | 1.452757 | 0.623361 | 0.140692 | 1.17% | 19.79% |

All rows have 1,024 questions. Overall Accuracy equals four-field macro Accuracy
because each field has 256 questions. Every field's NLL/Brier/ECE and mean
confidence, as well as both models' per-field Accuracy, are retained in JSON.

| Candidate condition | Color | Texture | Count | Shape |
|---|---:|---:|---:|---:|
| known | 100.00% | 100.00% | 100.00% | 100.00% |
| question | 42.19% | 89.45% | 95.70% | 21.09% |
| options | 100.00% | 99.61% | 53.52% | 100.00% |
| joint | 76.17% | 44.92% | 5.47% | 70.31% |
| distractor | 0.00% | 6.64% | 11.33% | 37.50% |
| distractor_options | 0.00% | 9.77% | 0.78% | 27.34% |
| no_question | 16.41% | 52.34% | 0.00% | 31.25% |
| reverse_joint | 76.17% | 44.92% | 5.47% | 70.31% |

The research gate requires BOTH `joint` and `distractor_options` to reach >=95%
for each field, >=90% pair-both Accuracy, >=3 points improvement over parent with
positive state-bootstrap lower bound, lower NLL, Brier no worse and ECE increase
<=0.02. `known` must remain >=99%. Seven recorded failures are:

- joint: minimum field Accuracy, pair-both Accuracy;
- distractor_options: minimum field Accuracy, pair-both Accuracy, improvement,
  positive paired lower bound and NLL.

No quality failure was used to interrupt or modify inference. The candidate's
new-description exclusion arm has mean confidence 29.08% versus 56.80% for parent,
so its ECE improves while its Accuracy deteriorates; this is not adoption evidence.

### Uncertainty and integrity controls

Candidate minus parent, whole-state paired bootstrap, 10,000 draws, seed 17,
percentile 95% with linear quantiles:

| Primary arm | Accuracy difference, pp [95% interval] | NLL difference [95% interval] |
|---|---:|---:|
| joint | +14.551 [+11.719, +17.383] | -0.337914 [-0.364160, -0.312740] |
| distractor_options | -13.086 [-15.137, -11.035] | +0.009191 [-0.067609, +0.085392] |

All eight intervals are in JSON. The parent/candidate were fixed before these
new rows were scored, but the candidate was previously selected on development
validation. This is one training seed and a newly specified development family,
with no global multiplicity, broad OOD, or sealed-audit claim.

Reversal preserves native predicted IDs in both models. Maximum probability
change after ID alignment is 2.22e-16 for parent and 3.33e-16 for candidate.
No-question distributions are exactly identical across the four labels for every
state. Neither model can answer a distinct-condition pair correctly without a
question. Tied native argmax behavior is explicitly tested; reordering arrays
before argmax must not hide a changed selected ID.

### Posthoc error localization from saved scores

| Condition | Full 40-option Accuracy | Correct field selected | Correct value within requested field |
|---|---:|---:|---:|
| known | 100.00% | 100.00% | 100.00% |
| question | 62.11% | 62.11% | 100.00% |
| options | 88.28% | 88.28% | 100.00% |
| joint | 49.22% | 49.22% | 100.00% |
| distractor | 13.87% | 15.14% | 95.21% |
| distractor_options | 9.47% | 12.99% | 82.52% |
| no_question | 25.00% | 25.00% | 100.00% |
| reverse_joint | 49.22% | 49.22% | 100.00% |

The last column deliberately uses the known requested field to restrict the
argmax to its ten candidates. It is a diagnostic, not a change to the 40-option
benchmark or an input supplied to a deployed model. It adds no model evaluation.
Under `joint`, wrong answers are entirely field-routing failures. Exclusion
instructions additionally impair within-field value ranking. This narrows the
next supervision question without proving a fundamental capacity limit.

The prior binding TRAIN corpus contains 2,048 rows but only eight instruction
strings. This supports investigating semantic variety and contrasting roles,
while preserving replay/retention, before extending the same training schedule.

## Actual execution and safeguards

| Stage | Seconds | Pair presentations | Peak allocated CUDA bytes | Peak host bytes |
|---|---:|---:|---:|---:|
| smoke | 12.926 | 5,120 | 315,492,352 | 1,221,468,160 |
| primary | 239.795 | 655,360 | 322,896,896 | 1,296,392,192 |
| fresh_replay | 243.417 | 655,360 | 322,896,896 | 1,297,334,272 |

Total recorded process-body time is **496.138 seconds (8.27 minutes)** and
1,315,840 candidate-pair presentations including smoke and full replay. These are
not measured GPU-busy seconds. Compute is local, cloud cost zero. Primary starts
2026-10-15 04:50:07.914000 UTC and ends 04:54:07.715748 UTC; replay ends
04:58:59.105938 UTC. Both complete normally, with no numerical/resource stops.

Windows, RTX 3080 Laptop 16 GiB, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4,
Transformers 5.6.2, NumPy 2.4.6; FP32 SDPA, deterministic Torch, TF32 disabled,
dropout disabled and four CPU threads. Exact package versions and effective
config are retained. One model is resident at a time. No environment upgrade.
Maximum real input length is 73 tokens, no truncation.

A separate two-TRAIN-state smoke first checked all instruction paths. Its known
interface exactly reproduced saved parent step-0 and candidate step-512 fixed-
TRAIN prefixes. It never scored new transfer states. The full comparison then
ran once, followed by one fresh-process replay: **50 deterministic payloads**
(scores, probabilities, per-question predictions, metrics and effective config)
match byte-for-byte. Model tensor hashes before/after every condition match;
no parameter requires gradients and none receives gradients. All source/code
pins remain unchanged. Low transfer scores are a quality failure, not an
execution failure.

An independent CPU audit recomputed all 16 summaries, 16,384 prediction records,
all four/pair metrics, all eight bootstrap intervals and all gate failures.
It verified cohort exclusion, target mapping, frozen digests, counts and
controls. Pre-execution review caught missing taxonomy pinning, absent full
no-question distribution comparison, and native tie-breaking masking; all were
fixed and regression-tested before corpus creation or GPU execution.

## Reproduction and artifact locations

From `C:/CodexWorktrees/pretrained-reuse-omni`, after checking the pinned inputs:

```powershell
python -m scripts.prepare_binding_transfer --output C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-corpus-01
python -m scripts.evaluate_binding_transfer --smoke --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-corpus-01 --corpus-sha256 b784233ee4dac03aaaf1be67aae97ed07b0dadc55e45f76a2fa2fbe5fc6634e4 --output C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-smoke-01
python -m scripts.evaluate_binding_transfer --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-corpus-01 --corpus-sha256 b784233ee4dac03aaaf1be67aae97ed07b0dadc55e45f76a2fa2fbe5fc6634e4 --output C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-eval-01
python -m scripts.evaluate_binding_transfer --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-corpus-01 --corpus-sha256 b784233ee4dac03aaaf1be67aae97ed07b0dadc55e45f76a2fa2fbe5fc6634e4 --output C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/binding-independent-transfer-eval-01 --replay-sha256 78d4982a053eb9db1714113e31bac5e1fc131a17396cda7936d800bbb3bb2da7
```

These are the actual commands, not permission to overwrite: runners reject
existing directories. Reproduction needs distinct output paths and the resulting
corpus/reference hashes; do not weaken provenance checks. Old checkpoints and
corpora stay untouched. New scores and prediction files remain outside Git.

| Artifact | SHA-256 |
|---|---|
| binding-independent-transfer-corpus-01 | `b784233ee4dac03aaaf1be67aae97ed07b0dadc55e45f76a2fa2fbe5fc6634e4` |
| binding-independent-transfer-smoke-01 | `42af0bc04b49822b9dd349977018fb5aff9c24138223a1d8db919b2124073cf8` |
| binding-independent-transfer-eval-01 | `78d4982a053eb9db1714113e31bac5e1fc131a17396cda7936d800bbb3bb2da7` |
| binding-independent-transfer-replay-01 | `0462cf5ea96d03ec3ed47749e8d3fb1bb223ebf4eb16bbb5876f60c3ea702afe` |
| binding-independent-transfer-field-diagnostic-01.json | `6c032efafaa9dbbf0da0e5618f9dfb80a45a5cda56c2016a21a3ce684284c71d` |

Local `binding-independent-transfer-publication-01.json` SHA-256:
`e3c658a8cc962f8312fea061b8b5332bd5f49be29aba7b10356e62b399261752`.
The tracked JSON is a content-equivalent copy; Git newline conversion may change
working-copy bytes. It bundles the checked results, full metrics and explicitly
posthoc diagnostic. The runner itself reproduces all scored evidence.

Local verification: Ruff passes, CPU pytest **1,595 passed / 1 skipped** in
22.90 seconds, and all **25 manifest/catalog commands pass**. New regression tests
cover deterministic exclusion, complete targets and option identities, both
primary quality gates, question-free distribution equality and native argmax ties.
The six new helper tests run without GPU/model downloads.

No new training, saved weights, main merge, audit access, quantization or product
promotion. Native conditional perception, temporal/joint evidence, integrated
<=200M runtime quality and four-modality 95–98% remain open.
