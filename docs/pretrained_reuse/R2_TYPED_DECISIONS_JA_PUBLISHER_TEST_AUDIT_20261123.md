# Japanese typed-decision publisher-test audit — 2026-11-23

## Purpose and limits

After selecting frozen readouts on the fixed train-only validation split, we
ran a **single, no-selection evaluation** of those unchanged checkpoints on
the previously unopened publisher `test` partition. This is a public
development-source audit, not the project's sealed audit. Test results must
not be used for another checkpoint, architecture, or hyperparameter choice.

The targets are probability distributions from the dataset's English-state
teacher model, not independently adjudicated ground truth. Metrics therefore
measure agreement with that teacher. This is a Japanese text-only typed-option
task and says nothing about image/audio/video or multimodal fusion quality.

## Frozen inputs and integrity

- Dataset: [`GeneLab/typed-decisions-ja`](https://huggingface.co/datasets/GeneLab/typed-decisions-ja), revision `01a4558e83bf5a748094239df843d658b33d2533`, declared Apache-2.0.
- Test parquet: `all/test-00000-of-00001.parquet`, 238,560 bytes, SHA-256 `8a8257b0aeb7b472c1cdf17f1150fd216f5cf7da2c157997270ef154e49de6d3`.
- Reference publisher train parquet: 633,166 bytes, SHA-256 `14589163a2278225926e599a24f5620902bbbdc998946d1f63fc716a4477be1c`.
- Train/test contain 1,200 / 400 unique state IDs. Normalized state JSON and state-ID overlap are both zero. Test contains 100 states per workflow and 2,000 typed decisions total.
- The test file's SHA matches the pinned Hub LFS object. It was fetched from the revision-pinned official dataset path after the two candidate readouts and selection rule were frozen.

## Models and fixed inference

| Candidate | Encoder revision / weights SHA-256 | Encoder parameters | Total incl. readout | Readout SHA-256 |
|---|---|---:|---:|---|
| Japanese BERT-base v3 | `tohoku-nlp/bert-base-japanese-v3` @ `65243d6e5629b969c77309f217bd7b1a79d43c7e`; `e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4` | 110,616,576 | 110,715,137 | `962c8e2f61694f38d9e55ae3befba4d44624009e7e1a3917d14adc505fabed8e` |
| LightSPLADE-derived Japanese BERT | `bizreach-inc/light-splade-japanese-28M` @ `64d8f31a2196074dbd5de7707bbe6f14be3ab985`; `4354b4cf64ca2a18eb60b38165a913365e74704457b6ad3f28631cebd769d0e1` | 27,566,592 | 27,616,001 | `e10043e9b12682a60440387b0a597a2084f856849366eedb59320b23bc3e62e5` |

Both readouts are the validation-selected epoch-8 artifacts from the prior
matched-protocol report; no fitting, calibration, prompt changes, or
checkpoint selection occurred here. Both use the same frozen-encoder CLS
features, Japanese BERT tokenizer, pair serialization, 512-token cap, and
`only_first` truncation. Each candidate encoded the same 7,100 option pairs;
2,201 (31.0%) exceeded the cap before truncation. This material truncation
limitation applies to both.

Execution used local CPU only, four Torch threads, Python 3.11.9, PyTorch
2.6.0+cu124, and Transformers 5.6.2. BERT evaluation took 2,081.77 seconds;
LightSPLADE took 492.64 seconds. No GPU or cloud compute was used. The
previous train-only report records BERT's 4.42x feature-extraction cost on
the larger train+validation corpus.

## Results

Macro values average the 12 workflow/question-type cells equally. Accuracy is
against the teacher's modal label; NLL and Brier use its full soft
distribution. ECE-15 compares confidence with the teacher probability of the
predicted option.

| Candidate | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE-15 |
|---|---:|---:|---:|---:|
| BERT-base | **0.5254** | **1.0639** | **0.1712** | **0.0506** |
| LightSPLADE-derived 27.6M | 0.4417 | 1.1299 | 0.2025 | 0.0587 |

Paired state-cluster bootstrap (10,000 PCG64 replicates, seed `20261123`)
resampled 100 state IDs within each workflow and retained all five questions
for each selected state. BERT minus LightSPLADE differences were:

| Metric | Difference | Percentile 95% interval | Fraction of resamples favoring BERT |
|---|---:|---:|---:|
| Macro Accuracy | +0.0837 | [+0.0587, +0.1083] | 100% |
| Macro NLL | -0.0659 | [-0.0730, -0.0590] | 100% |
| Macro Brier | -0.0312 | [-0.0352, -0.0274] | 100% |
| Macro ECE-15 | -0.0063 | [-0.0188, +0.0064] | 84.36% |

The bootstrap is descriptive uncertainty for this already-opened test only;
it does not undo validation checkpoint selection, the teacher-label
limitation, translation issues, or establish product quality. ECE's interval
includes zero.

### Per-cell Accuracy and NLL

| Workflow / question type | N | BERT Accuracy | Compact Accuracy | BERT NLL | Compact NLL |
|---|---:|---:|---:|---:|---:|
| agent trace / choice | 200 | 0.290 | 0.310 | 1.340 | 1.360 |
| agent trace / noul | 100 | 0.480 | 0.450 | 0.700 | 0.710 |
| agent trace / score | 200 | 0.310 | 0.260 | 1.350 | 1.420 |
| customer service / choice | 200 | 0.660 | 0.350 | 1.250 | 1.530 |
| customer service / noul | 100 | 0.660 | 0.650 | 0.670 | 0.680 |
| customer service / score | 200 | 0.430 | 0.240 | 1.220 | 1.330 |
| invoice processing / choice | 100 | 0.450 | 0.360 | 1.250 | 1.290 |
| invoice processing / noul | 200 | 0.720 | 0.720 | 0.520 | 0.590 |
| invoice processing / score | 200 | 0.420 | 0.380 | 1.230 | 1.290 |
| security incidents / choice | 100 | 0.740 | 0.740 | 1.230 | 1.260 |
| security incidents / noul | 200 | 0.650 | 0.540 | 0.680 | 0.690 |
| security incidents / score | 200 | 0.480 | 0.290 | 1.320 | 1.400 |

## Decision

This held-out result strengthens the case that **BERT's contextual CLS
representation transfers better than this compact retrieval-derived CLS
representation** on this typed Japanese task. It does not establish that BERT
is a generally better Decision encoder: the targets are teacher-generated,
31% of pairs are truncated, and task coverage is four synthetic workflows.
The 110.7M BERT encoder plus readout also consumes more than half of the
project's provisional 200M pre-quantization budget before any image, audio,
video, fusion, or sensor components. Its 52.54% macro teacher-mode accuracy
is far below the project's 95–98% aspiration.

Keep BERT-base as a teacher/reference. Do not rerun this test or tune on these
results. The next BERT-family candidate should be a single compact
sentence/option encoder or distilled student, evaluated on a newly admitted
grouped source with human-adjudicated or otherwise independently justified
targets and complete resident-parameter accounting. This result remains
text-only evidence and does not close R2's multimodal requirement.

## Reproduction artifacts

The test parquet, complete machine-readable metrics, predictions, and
bootstrap output are retained outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\typed-decisions-ja-publisher-test-audit-20261123-attempt-01\`.

| Artifact | SHA-256 |
|---|---|
| Test evaluation report | `1becbc8f28b678c339409b1e52f44452f6c2dd72085636cdf034f3425569adfe` |
| BERT per-example predictions | `305a54a38bd157ed67e12babf2a54f15d90693ec6a655d08dfd3157764a58b0c` |
| Compact per-example predictions | `0299b54233d741610d776b7c459f6ab38e8436597a56652c8ca5713162dbcd63` |
| Paired bootstrap JSON | `163ff990dbdaafa800c183595be9fb3b5a4a6b903e3daba42b3a43859f7f43bae` |
| Test-only evaluator source | `6d9f6d5903d006c0e09ccd6d2795d6d4bae6194078b1866fe385ed06cad791c6` |
| Bootstrap analysis source | `840c6880f6f15e5bf3ee28ae146466803f7421e6345023ddfe96d2eceebcc337` |

The evaluator has no optimizer or selector path. It explicitly accepts only
the publisher test split, verifies exact state-ID and normalized-state
disjointness from publisher train, checks candidate alignment, and records
every per-example probability. The analysis script consumes saved predictions
only; it does not reload media or models.
