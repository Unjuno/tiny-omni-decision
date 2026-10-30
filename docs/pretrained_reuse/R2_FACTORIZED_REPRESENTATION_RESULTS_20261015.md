# Conditional representation and factorized readout: measured result

Execution **PASS**, research continuation gate **FAIL for both models**. No model
adoption, fitting, quantization or sealed-audit access. The goal of a native
four-modality <=200M model with each required task at95–98% remains unmet.

[Prospective protocol](R2_FACTORIZED_REPRESENTATION_PROTOCOL_20261015.md) and
[complete metrics, environments, commands and hashes](R2_FACTORIZED_REPRESENTATION_RESULTS_20261015.json).
Frozen source commit: `81b6f8f42cb99fa124e815f9e789f41144392f38`.

## What ran

Two existing fixed checkpoints, each with direct and factorized readouts, followed
by candidate reversal, donor-state and no-request controls. Every decision keeps
40 supplied options. Factoring computes p(field|request) * p(value|record,field).
The four-way routing score alone is never reported as 40-way Decision accuracy.
All parameters are frozen, FP32, temperature1, fixed batch32, no truncation.
No prompt/temperature/seed/checkpoint search or new training took place.

MiniLM retained KL512 has22,713,601 parameters. DeBERTa xsmall has70,830,722,
including49,191,168 embedding parameters. Reused pinned checkpoints and native
heads; no downloads or installed-environment changes. Their pretraining/history
differ. The common direct rendering also differs from the old MiniLM interface.
This measures complete interface alternatives, not an isolated capacity effect,
nor a reproduction of historical benchmark scores. Factorization also changes
input phrasing/length and normalization; their contributions are not isolated.

Excluded7,580 prior attribute-state identities. Used32 new combinations, eight
fixed syntax families, four requests per state/family:1,024questions,40,960candidate
presentations per primary direct arm. There are only32 distinct request strings
and32 states, not1,024 independent language situations. Values/vocabularies are
already known; no new natural concept, native media asset, or blind audit is used.
The count attribute is a string lookup, not a numerical-comparison achievement.

## Full-candidate primary results

| Model | Readout | Accuracy | NLL | Brier | ECE15 | Mean confidence | Pair both correct | All4 correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minilm | direct | 32.4219% | 2.015381 | 0.899847 | 0.324783 | 0.644313 | 8.2031% | 0.0000% |
| minilm | factorized | 28.9062% | 2.561509 | 0.871908 | 0.326471 | 0.599722 | 19.1406% | 12.1094% |
| deberta | direct | 34.0820% | 2.147589 | 0.817627 | 0.174354 | 0.166467 | 11.8490% | 1.9531% |
| deberta | factorized | 59.0820% | 1.032497 | 0.473600 | 0.118114 | 0.587512 | 42.5130% | 32.8125% |

All eight families (128questions each):

| Family | MiniLM direct Acc/NLL | MiniLM factorized Acc/NLL | DeBERTa direct Acc/NLL | DeBERTa factorized Acc/NLL |
| --- | --- | --- | --- | --- |
| explicit | 55.4688% / 1.285876 | 99.2188% / 0.111205 | 0.7812% / 2.516988 | 96.8750% / 0.230024 |
| irrelevant_first | 24.2188% / 2.256926 | 0.0000% / 2.565340 | 49.2188% / 1.932819 | 96.8750% / 0.298609 |
| positive_then_not | 34.3750% / 1.781486 | 32.0312% / 1.516448 | 34.3750% / 2.238244 | 96.8750% / 0.218718 |
| exclude | 12.5000% / 2.739361 | 0.0000% / 5.188464 | 47.6562% / 2.218465 | 25.0000% / 1.951994 |
| other_than | 13.2812% / 2.531694 | 0.0000% / 5.655995 | 37.5000% / 2.076821 | 25.0000% / 2.106940 |
| although | 29.6875% / 1.946710 | 25.0000% / 1.461328 | 53.9062% / 1.924799 | 25.0000% / 1.377812 |
| synonym | 57.0312% / 1.312781 | 75.0000% / 0.487814 | 12.5000% / 2.185504 | 75.0000% / 0.867435 |
| exclude_synonym | 32.8125% / 2.268210 | 0.0000% / 3.505476 | 36.7188% / 2.087075 | 32.0312% / 1.208445 |

Per-family/per-field Brier/ECE and all saved probabilities are retained in the JSON/artifacts.

## Paired uncertainty and interpretation

- minilm: factorized minus direct Accuracy -3.515625pp,95% interval [-4.882812,-2.148438]; NLL +0.546128, interval [+0.445494,+0.641895].
- deberta: factorized minus direct Accuracy +25.000000pp,95% interval [+21.582031,+28.906250]; NLL -1.115092, interval [-1.136988,-1.094167].

Paired whole-state bootstrap:10,000resamples, seed17, percentile95%, linear
quantiles. All questions for a state stay together. These intervals do not resample
syntax families and cannot establish generalization to arbitrary language.

DeBERTa improves aggregate Accuracy/NLL/Brier/ECE with factoring, but exclusion,
other-than and although families remain25%; some families worsen versus direct.
MiniLM factoring worsens aggregate Accuracy/NLL, despite a small Brier improvement.
Thus neither meets the predefined minimum-family90% and retention requirements.
Do not describe the DeBERTa gain as universal capability or dismiss it because
the full gate fails. Record both facts.

The no-request control intentionally has four different requested truths for the
same state with no identifying request; its roughly25% does not require a textual
front end for native product tasks. Native tasks with a fixed decision specification
remain valid. Oracle structured states here do not prove native perception quality.

## Separate factors: descriptive post-hoc diagnostic

The field/value split helps localize this particular failure. It is not another
selection rule or a new independent experiment. Repeated factors are not unique data.

| Model | Request-factor field Acc | Within-field value Acc | Value Acc by field |
| --- | ---: | ---: | --- |
| minilm | 31.250% | 99.219% | color 100.000%, texture 96.875%, count 100.000%, shape 100.000% |
| deberta | 59.375% | 96.875% | color 100.000%, texture 87.500%, count 100.000%, shape 100.000% |

This shows that matching the requested field remains weaker than looking up
values in these supplied records. It does not isolate model capacity from task
coverage, imperative/NLI mismatch, learned scoring bias, or input representation.
No claim that numerical reasoning, temporal reasoning or original tasks were retained.

## Controls and complete fresh replay

| Model | Readout | Donor truth Acc | Original truth after donor swap | No-request Acc | Reversal ID agreement | Max probability delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| minilm | direct | 32.4219% | 2.7344% | 25.0000% | 100.00% | 1.85e-06 |
| minilm | factorized | 28.9062% | 2.9297% | 24.2188% | 100.00% | 4.3e-07 |
| deberta | direct | 34.0820% | 3.4180% | 23.4375% | 100.00% | 6.92e-07 |
| deberta | factorized | 59.0820% | 5.2734% | 25.0000% | 100.00% | 4.44e-16 |

Donor accuracy is reported against both donor and original targets. Donor
states are a permutation of the same32 states, so unchanged donor-truth aggregate
accuracy is expected; it is not evidence on new states. Reversal executes new
forwards and compares actual predicted IDs, including tie behavior.

Both full fresh-process replays passed: each16raw-array files, eight prediction
files and metrics.json matched the primary byte-for-byte (25payloads/model).
Tolerance was1e-6 with exact discrete IDs/counts; observed replay differences were0.
Order-induced floating-point differences are a separate comparison from fresh replay.
Every checkpoint/input/code hash remains intact. An independent CPU review
recomputed the complete primary outputs, interval, controls and accounting.

## Resources and artifacts

RTX3080Laptop16GB, local only, zero new optimizer updates and cloud cost0.
Each full model process requested344,064 pairs; exact within-call deduplication
left129,668 actual model pair forwards across all controls, including pairs repeated
between calls. There are43,652 globally distinct rendered text pairs; neither count
is an independent question count. Primary unique pairs:
direct40,960, request128, values1,280. Reversal/donor deliberately re-execute
forwards rather than reuse the primary result. Full replay executes the same work.

| Process | Seconds | Peak CUDA allocated bytes | Peak host bytes | Result SHA256 |
| --- | ---: | ---: | ---: | --- |
| factorized-condition-minilm-smoke-01 | 8.593 | 159049728 | 1237721088 | 609c0c5797e6242c4bce0496c457b904dec13f6a039e4032125be73ab12648d8 |
| factorized-condition-minilm-01 | 65.965 | 166479360 | 1296887808 | 5789860b0491a6dff724dbc444d1d9e04470dfbcf8540977a355c5a6c1b74bc2 |
| factorized-condition-minilm-replay-01 | 68.991 | 166479360 | 1295896576 | 4251e81fc0900b490670f255d580c7a4114cca4a14d7a077661075e688833599 |
| factorized-condition-deberta-smoke-01 | 10.986 | 454040576 | 1768366080 | 382e26fe545c071d80e26ee01397d4376359468f6abcbf75acf13efc556617d4 |
| factorized-condition-deberta-01 | 174.509 | 470305280 | 2015997952 | 747ca8c360824f031ad0127c86f2f10182bb4aac31b0d4c99fbfc4895cb88927 |
| factorized-condition-deberta-replay-01 | 175.857 | 470305280 | 1820839936 | 6ce5792cc5e2c7d2e49b3291be43a81a773807bb5b8962f1983bce2b4cf37e9f |

Sum of six measured process bodies: 504.903s (8.415min). Includes loading/I/O/inference, not GPU-kernel time. Electricity unmeasured.

Environment/actual attention backend, commands, UTC times and all payload hashes
are in the companion JSON and each run's effective-config.json/result.json.
Artifact root: `C:/CodexArtifacts/pretrained-reuse-r2/`, with distinct corpus,
smoke, primary and replay directories named in the table. Existing weights and
corpora are unchanged. No weight/media files are added to Git.

Corpus manifest SHA256: `d449339a5191daec8d370d47ca5bf8e0368ebcfcc37f08f222e447f9ea0454d4`.
Publication SHA256: `d811e9e072e504808c1ed7e5c0d627fa6b11ac8c684d6e3c43f112bd010bcbc4`.
The one-shot CPU publication verifier is saved locally as
`C:/CodexArtifacts/pretrained-reuse-r2/publish-factorized-condition.py`; its hash
is in the JSON. It rechecks hashes, probabilities, factor composition and full replay.

## Budget and next action

Using measured counts, DeBERTa+VideoPrism+Whisper-encoder+environment-audio sums
195,365,529 before all temporal/fusion/readout weights; only4,634,471 remain.
Keeping full Whisper instead gives224,917,785 and fails the200M limit. Encoder-only
speech quality is unproven, so the arithmetic is not an admitted Omni architecture.
MiniLM's encoder-only scenario is147,248,408 before heads, with the same caveat.

Close this fixed zero-shot representation/rendering screen. Neither model is
selected for product or additional training from these scores. No prompt, mixture,
temperature, coefficient or seed sweep on these observed states/syntaxes.
Retain the genuine DeBERTa aggregate gain as evidence that the complete readout
interface matters; pretrained NLI plus factorization alone is insufficient here.

The next separate contract must cover conditional relevance, state consistency,
numerical decisions and original-task retention together, using independent
training/selection/transfer states and syntax. Define those labels and retention
coverage before another semantic fit. The old numeric/FLAN failures remain valid;
do not repeat their fixed256-question probe or treat field matching as arithmetic.
Architecture choice and a new training experiment remain unselected. A larger
language allocation requires whole-product quality/cost evidence. Keep native
input/JSON infrastructure and connect the same decisions to real modalities after
conditional reliability improves; temporal/joint/OOD/product gates remain open.

## Verification

Ruff passed; CPU suite1,646passed/1skipped (23.35s), targeted10passed; all25manifest
commands passed. The initial manifest invocation incorrectly used the package
instead of its CLI module; failure log01 is retained, corrected log02 passed.
Code-review findings (tie identity, numeric replay tolerance, missing source pin,
final resource guards) were fixed before corpus freeze and any model execution.
CI1761 succeeded for frozen source81b6f8f. Final report-commit CI is recorded in
`C:/CodexArtifacts/pretrained-reuse-r2/factorized-condition-results-ci-01.json` after push.
