# R2 Japanese BERT text reuse results — 2026-11-23

## Decision

Keep `tohoku-nlp/bert-base-japanese-v3` as a **strong Japanese text teacher and
research comparator**. Do not replace the text towers in the current four-tower
candidate with this BERT-base checkpoint: the one-for-one replacement exceeds
the whole-model 200M pre-quantization parameter target. The current evidence
supports Japanese text quality on two development tasks; it does not establish
95% quality, broad multilingual quality, or multimodal fusion quality.

This record audits already-completed CPU experiments stored outside Git. It
does not start a model run, change weights, or access any publisher test or
sealed audit data. The source-code commit for the MASSIVE run was not recorded
in its run report and remains **UNKNOWN**; executed scripts and their hashes
are preserved with the artifacts.

## Japanese 60-intent MASSIVE comparison

Both encoders use the same frozen seed-17 split: 11,228 train and 2,033
development-validation examples, all from the Japanese 60-intent task. The
split has zero cross-split source-ID overlap and zero normalized-utterance
overlap. The publisher test partition was not downloaded or opened. Both
checkpoints were selected on this validation set after eight epochs, so this
is development evidence affected by selection.

| Frozen encoder and readout | Total parameters before quantization | Encoder weight bytes | Accuracy | Macro Accuracy over 59 represented intents | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Japanese BERT-base v3; `[CLS] -> 128 tanh -> 60 logits` | 111,313,340 | 447,406,217 | 0.8165 | 0.7524 | 0.6872 | 0.2730 | 0.0360 |
| Ruri-v3-70m; mean pool + L2, same-size 60-way readout | 70,059,836 | 280,019,040 | 0.7550 | 0.6849 | 0.8983 | 0.3483 | 0.0559 |

The readout has 106,172 trainable parameters for BERT and 57,020 for Ruri.
The paired comparison resampled 2,015 normalized-utterance groups for 10,000
PCG64 replicates with seed 17. BERT minus Ruri Accuracy was +6.15 percentage
points (percentile 95% interval +4.28 to +8.06 pp); NLL difference was -0.2111
([-0.2654, -0.1577]) and Brier difference was -0.0752 ([-0.0951, -0.0556]).
The interval describes these validation predictions; it does not undo
checkpoint-selection exposure or turn validation into an independent test.

The best-epoch pattern is not uniformly better: BERT epoch 8 improved
sample-weighted Accuracy/NLL/Brier from epoch 7 but reduced macro Accuracy
from 0.7659 to 0.7524 and slightly worsened ECE from 0.0349 to 0.0360. This
supports retaining per-intent and calibration metrics in model selection.

Save/reload verification passed for both models on all 2,033 validation rows:
all argmaxes matched and maximum probability delta was exactly 0. The frozen
encoder was not fine-tuned; only the small readout was trained.

## Five-choice Japanese CommonsenseQA cross-task probe

This task asks the model to score question-option pairs, which is closer to a
variable-option Decision interface than the fixed 60-intent MASSIVE head. The
two runs share the 8,054-row train / 2,004-row development validation split,
with zero train-validation question-group overlap; the official test was not
opened. Validation selected epoch 7 for both.

| Encoder and scorer | Total parameters before quantization | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Japanese BERT-base; joint `[question, option]` `[CLS]` scorer | 111,305,729 | 0.7096 | 0.7904 | 0.4047 | 0.0383 |
| Ruri-v3-70m; pair-feature scorer over `[q,c,|q-c|,q*c]` | 70,199,681 | 0.5070 | 1.2193 | 0.6129 | 0.0292 |

This is supporting evidence only. The scorer architectures differ, so the
large gap cannot be attributed to the encoder family alone. The model reports
record selected validation metrics; no independent test result is available.

The shared split hashes are train
`358dde70ee3b2e80fc70e1a09c4dc69d5e956079bc2444e4189adb79fba92de6`,
validation `3e2a71524e0aeb789141f0591df361ee4a5edf889a469bbe595a0483437a9f61`,
and manifest `af9b142c96b3aa9e62fb4eedd426e145ccfc069d282f29f5abad34a864d4beea`.
There are 7,938 train and 1,984 validation question groups with zero overlap.
The source reports CC-BY-SA-4.0; no official test records were opened. BERT
readout extraction/training used CPU for 905.19 / 3.17 seconds; Ruri used CPU
for 299.66 / 4.33 seconds. These are observed experiment timings, not
end-to-end package runtime benchmarks.
The BERT run's source-code commit is `4476fe1abceaa993c82200301fd2658393db728e`;
the Ruri run's source-code commit is `6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33`.

## Three-way Japanese NLI probe (existing artifact audit)

A third, already-completed frozen-encoder probe was present in the local
artifact store and is included here for completeness. It uses the pinned
Japanese BERT-base v3 checkpoint and a 128-hidden tanh pair scorer on
premise/hypothesis inputs. The grouped development split has 18,005 train and
4,502 validation pairs across 8,097 normalized-sentence/image groups, with
zero group, image, or normalized-sentence overlap. The publisher test was not
opened. The selected epoch-6 readout has 111,305,987 total parameters before
quantization and reached 78.10% accuracy, 0.5592 NLL, 0.3156 Brier, and
0.0199 ECE-15. Accuracy is badly uneven across labels: entailment 24.18%,
contradiction 78.44%, and neutral 91.00% (macro class accuracy 64.54%). The
training CE continued falling through epoch 8 while validation NLL was best
at epoch 6 and worsened at epoch 7, so this probe also shows a generalization
gap and class-imbalance weakness. It is not a supplied-choice benchmark and
does not support a 95% claim.

The source is the pinned JGLUE repository revision
`6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33` (CC-BY-SA-4.0, as recorded in the
split manifest). This license and its share-alike obligations need separate
product review before any derivative dataset use. The BERT weights, split,
feature cache, readout, predictions, and training script were rehashed against
the external run report; every recorded hash matched. The run was CPU-only,
seed 17, eight epochs, with validation-NLL selection. Its source commit was
`4476fe1abceaa993c82200301fd2658393db728e`; this experiment must not be
repeated or retuned on the exposed validation split.

This is evidence that a Japanese BERT pair scorer can emit three-way
probabilities on one grouped NLI development set, but its poor
entailment recall makes it a weak direct option-verification mechanism. It
does not change the whole-model budget decision or replace the need for a
fresh, task-admitted Decision evaluation.

## Whole-model budget impact

The pinned R1 four-tower proposal (V-JEPA 2.1 ViT-B encoder + Japanese
Ruri-v3-30m + English MiniLM-L3 + Whisper Tiny) totals 178,689,152 encoder
parameters before its Decision readout. Replacing the 36,705,536-parameter
Ruri-30m encoder with the 111,207,168-parameter Japanese BERT-base encoder
would make that stack 253,190,784 encoder parameters, **53,190,784 above**
the 200M ceiling, before counting any Decision head. Adding the measured BERT
60-way head would raise this to 253,296,956, though that fixed-task head is
not itself a general option scorer.

The BERT encoder weight file alone is 447,406,217 bytes in the measured
artifact. The R1 stack's provisional package target applies after actual
quantized export and reload; no BERT quantization was run here. Parameter
count and stored bytes are separate gates.

## Reproducibility and artifact locations

Artifacts are retained under
`C:\CodexArtifacts\pretrained-reuse-r2-local\` and are intentionally not
copied into Git:

- MASSIVE BERT: `massive-ja-jp-bert-readout-20261123-attempt-01/`
- MASSIVE Ruri: `massive-ja-jp-ruri70m-readout-20261123-attempt-01/`
- MASSIVE paired comparison: `massive-ja-jp-bert-ruri70m-paired-comparison-20261123-attempt-01/`
- CommonsenseQA BERT: `jcommonsenseqa-tohoku-bertbasev3-readout-v1-attempt-01/`
- CommonsenseQA Ruri: `jcommonsenseqa-ruri70m-readout-v1-attempt-01/`
- JNLI BERT: `jnli-tohoku-bertbasev3-readout-v1-attempt-01/`
- JNLI grouped split: `jnli-yjcaptions-grouped-split-v1-attempt-01/`

MASSIVE data hashes (BERT and Ruri runs): train
`ab55ddd41e4cf0f07f65bddc84432bea40ceb36caaa94c6edf15499ddf6a1270`,
validation `aa42d55cb8afcf7170483322e6a129acd4657a9acf3ea3530fea265a36679707`,
split manifest `d9fdcc0235b3caecac462574e6111a6c1bf59102eab3ce39361c7fb42af9ed1d`.
The BERT model is pinned to revision
`65243d6e5629b969c77309f217bd7b1a79d43c7e`; its measured weights SHA-256 is
`e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4` and the
run records Apache-2.0. Ruri is pinned to
`07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc`; weights SHA-256
`c959b76a535eee838b1f1e6b8999ba513ef8d87a6f8c3534c50dcda1eb5f6a62`.

| Artifact | SHA-256 |
|---|---|
| MASSIVE BERT `run-report.json` | `6f5842e2d12b87585dc937305a55824191ec90fbf29041a8eb85a42c690b0312` |
| MASSIVE Ruri `run-report.json` | `53339d55671dcd1c5541f9b948d4587bec50a3b8fecbe0cc8ca56642bd928076` |
| MASSIVE paired `comparison.json` | `2fdf545b4cdf854e059b9c9f835242140143850b3c4b5e6233dd104f61e0eb6c` |
| MASSIVE BERT `best-readout.pt` | `49d22f13ac6a4b4f3f31798d70d986d815950eb4e4460cb30cc34ab7e012254b` |
| MASSIVE Ruri `best-readout.pt` | `1cae7a5015b134e0061c2f7f3b75759bc0ff4347e1fcacaddec2e3edf29033a4` |
| MASSIVE BERT saved validation predictions | `8c675c3afe1893bec47b53214678935964d4e06931e4a3698a3abc2772319040` |
| MASSIVE Ruri saved validation predictions | `f6e53c1524b80ee8d0ea03368e14b5707054ac27f8eff160530d39546627f2ed` |
| BERT training script | `e2e7a51d90c2f308528b23c3ccf6c10bcc1c809118b0a773e330701d9441131e` |
| Ruri training script | `b273e42de2c47533433a0ba9f1e4e1b29c53f362494168326efc6a8049ae698d` |
| CommonsenseQA BERT `run-report.json` | `945d6d1413fcadae3941bb94fb5bce8604d7bb023d5cf9f5d9e481f6da51a140` |
| CommonsenseQA Ruri `run-report.json` | `0866a25ad93eb3e4e1313e79bd717edf87b055dd76f831a37527b187ffd8f8ad` |
| CommonsenseQA BERT best readout | `e5936a7c86c2a78ac49fb0e386cc666ee02417fa5e7af793e4dd38c0c6f80e31` |
| CommonsenseQA Ruri best readout | `0f00851f2f6268b2e1c9373c16a0fbde5431ce7a6429af3f0ef0d39538784aad` |
| CommonsenseQA BERT predictions | `ba409e600a190c2e2468a13e5eb546af64b7fee9d3d4a45c8e9ce1be7d5334dc` |
| CommonsenseQA Ruri predictions | `2044d741c9d165728d502f73f0c6fe1675b51496c0a505ed3dae7875f21fb5f2` |
| CommonsenseQA BERT training script | `f4f0239da44fcfbbc4c9255856825060305e5ed1520ce033752e03a53c3bdae2` |
| CommonsenseQA Ruri training script | `566367e4ac23c7280dbf77fe51ff503a91bd388b04eed8346d6e51aab9970d34` |
| JNLI BERT `run-report.json` | `e21f98c3f55b028a5eeca2e5176e5830cce598d70d38ec07a77c9564ee6d076b` |
| JNLI BERT `best-readout.pt` | `d59e5cc36795cf9e50bd2b864f3de048c27fa9a6de67cf428c7ef72aea2de91b` |
| JNLI BERT frozen features | `8fd40588ff16dbc1cf0fba39b316a96a7c4837ad0fdedb61e3712c9e76b5c038` |
| JNLI BERT validation predictions | `8c0d07f74c1ab070ad1b8c969426198e5eeb30c1c91a3c114e19ad750f0fc3b0` |
| JNLI BERT training script | `37827c8567cd47b535b1b7fe96bffe077456f68555595ea52fd9d9cdf30c4833` |
| JNLI train split | `cca3b23dc2c8b4bd2d340d697163ce262d2310bad82ecdb95258c06d7a8d5d6c` |
| JNLI validation split | `253cd3234430d4bce79e287a38b317a936534bdd65c89bfbfb1524785b4b7aaa` |
| JNLI split manifest | `cbf1e92f74ef042cf019514627ac221af9a7c6a8068c118c4697ccc7b83c8156` |

The MASSIVE runs record Python 3.11.9, PyTorch 2.6.0+cu124 and Transformers
5.6.2, but ran on CPU; CUDA was not used. The separate Orbitune GPU process
was observed live during this audit and was left untouched. No new GPU work,
cloud compute, quantization, or product promotion occurred.

## Next decision

Do not repeat these BERT/Ruri comparisons on the already-observed validation
sets. Retain BERT-base as a quality teacher/reference, not a direct four-tower
replacement. If a text improvement is still needed after R2's frozen modular
baseline, the budget-compatible next experiment is a **single** predeclared
student path (for example, distilling BERT option probabilities into a
sub-40M Japanese text encoder) evaluated on a newly grouped development
source. This is conditional on data admission and must be compared against the
existing Ruri/MiniLM path under the complete resident-parameter accounting.
It does not supersede R2's need to establish direct modality and genuinely
joint-input Decision quality.

## Compact BERT-family candidate (metadata screen only)

The official Hugging Face model card for
[`bizreach-inc/light-splade-japanese-28M`](https://huggingface.co/bizreach-inc/light-splade-japanese-28M)
declares Apache-2.0, 27,747,968 FP32 parameters, an 8-layer/384-hidden
Japanese BERT architecture, and Japanese mMARCO sparse-retrieval training.
The Hub metadata revision observed on 2026-11-23 is
`64d8f31a2196074dbd5de7707bbe6f14be3ab985`. This is small enough to replace
the 36,705,536-parameter Ruri-v3-30m tower with about 8.96M fewer parameters.
No weights were downloaded and no quality run was performed. Its retrieval
training and sparse-vocabulary output make it a candidate to screen, not
evidence that it will improve option scoring; it should only be tested on a
newly admitted, grouped development source with a fixed scorer and the Ruri
baseline retained. Official model card claims and intended use can be reviewed
at the linked card; the revision and size above came from the Hugging Face
model metadata API. This metadata check does not establish downstream
redistribution or product suitability.

## Compact candidate CPU runtime audit

Downloaded only the pinned model files to the new external artifact directory
`C:\CodexArtifacts\pretrained-reuse-r2-local\light-splade-ja28m-runtime-audit-20261123-attempt-01\`.
At revision `64d8f31a2196074dbd5de7707bbe6f14be3ab985`, the 111,007,808-byte
`model.safetensors` hash is
`4354b4cf64ca2a18eb60b38165a913365e74704457b6ad3f28631cebd769d0e1`, matching
the revision-pinned Hub metadata. A CPU-only `AutoModel` load with the unused
MLM/SPLADE vocabulary head and pooler omitted produced finite last-hidden
states of shape `[1, 26, 384]`. This encoder path contains 27,566,592
parameters; the publisher's complete SPLADE checkpoint declares 27,747,968.
On Python 3.11.9 / PyTorch 2.6.0+cu124 / Transformers 5.6.2 with four CPU
threads, observed load time was 0.818 s and first forward time was 0.060 s.
CUDA was hidden from the process. The tokenizer initially failed because
`fugashi` was absent; pinned `fugashi==1.5.2` and `unidic-lite==1.0.8` were
installed only under this external artifact directory, leaving the shared
Python environment unchanged. This is a runtime smoke, not a quality result.
The script, Hub metadata snapshot, and machine-readable runtime report are
preserved in that directory; no weights or media were added to Git. Their
SHA-256 hashes are: script
`a0ae93155fe64f2a82225c60320d3c0fa03e4e15138fd900e9e57d34fb1c36ea`, runtime
report `17478a5b260a254de0829ca4708c025dd414b9f7eb35fb2e503463d764d58238`,
and Hub revision metadata
`d43bb05e2033fec3c6bd34efc820756be811407697f3a1747f833db7ba78c3dd`.

## Follow-up: typed-decision quality result

The metadata/runtime screen above was followed by a frozen-feature comparison
on a train-only Japanese typed-decision split. The 27.6M LightSPLADE-derived
CLS encoder scored 43.40% macro teacher-mode Accuracy, 1.1300 soft NLL, and
0.2282 Brier; same-condition 110.7M Japanese BERT-base scored 53.40%, 1.0751,
and 0.2010. A paired state bootstrap supports the validation difference, but
both readouts were selected on that validation and the targets are English-
state teacher distributions, not human truth. The full experiment, per-task
results, uncertainty, hashes, truncation, and compute cost are documented in
[the typed-decision comparison report](R2_TYPED_DECISIONS_JA_BERT_COMPARISON_20261123.md).
