# Physical Concepts Path A transfer result (2026-11-27)

## Decision

The frozen VideoPrism-B plus next-feature LSTM did **not** transfer reliably to
the fixed possible/impossible probe task. Across the five probe categories,
quadruplet ranking accuracy was 0.3875 (95% group-bootstrap interval
0.3770–0.3980; chance is 0.5). The clip-level anomaly threshold was highly
one-sided: sensitivity 0.988, specificity 0.039, and ROC AUC 0.543. This is a
negative result for this fixed representation, predictor, score, and source
transfer. It does not establish that temporal models, VideoPrism, or the
broader model family cannot learn video decisions.

The result points more strongly to **task/data mismatch in the learned score**
than to a corrupt source file: source generations, file sizes, MD5/SHA-256,
TFRecord CRC/schema, selected group identities, and exact media overlap gates
passed. The freeform training data used a next-feature prediction objective and
moving cameras; the evaluation probes use static-camera, synthetic
possible/impossible physics judgments. Low quality here is not fixed by adding
more BERT capacity. BERT remains a text-only, task-specific candidate.

## Frozen experiment

- Source repository: Google DeepMind Physical Concepts at
  [`baa2776017ea3cc8c873ce682bb6646860e12cae`](https://github.com/google-deepmind/physical_concepts/tree/baa2776017ea3cc8c873ce682bb6646860e12cae),
  dataset materials stated as CC BY 4.0 in the pinned source.
- Probe manifest: [`physical-concepts-probe-512-eval-20261127.json`](../../manifests/pretrained_reuse/physical-concepts-probe-512-eval-20261127.json),
  SHA-256 `6de6bcfda4c5406bffc211d7ddbfc5d06f497c393b6d2c9a05db3a2e8944b0ce`.
- Candidate: VideoPrism-B `google/videoprism-base-f16r288`, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`; upstream source commit
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; encoder checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Frozen predictor: seed 17, 64-wide LSTM, selected epoch 66 by freeform
  validation cosine loss. Predictor checkpoint SHA-256
  `40e9fd86aeb384f30459c346e3a5288cc6a7281ccf1e68854b9001bf44af8e49`;
  validation run metadata SHA-256
  `331a10ebd021553b54f08378eb66c234013882252ffe793d7f4f7b7214ed8ede`.
  Its validation loss was 0.0026964 versus 0.0080918 for persistence.
- Sample rule: immutable manifest rank, seed 17, 512 complete groups in each
  of Continuity, Solidity, Object Persistence, Unchangeableness, and Directional
  Inertia. The sample contains 2,560 groups / 10,240 clips, 10.24% of each
  category. It was chosen without labels, model scores, or predictions.
- Scoring was frozen in
  [`R2_PHYSICAL_CONCEPTS_PROBE_PROTOCOL_20261127.md`](R2_PHYSICAL_CONCEPTS_PROBE_PROTOCOL_20261127.md)
  before probe payload access. Each clip uses mean cosine prediction error over
  seven feature transitions; each group compares mean impossible-clip surprise
  with mean possible-clip surprise. Ties count one half. The one-class q95 and
  MAD scale were frozen from freeform validation before probe payload download.
- The pinned VideoPrism temporal representation uses non-causal full-clip
  attention. The LSTM target features may therefore contain information from
  future frames. Interpret this as a frozen feature-sequence anomaly-transfer
  diagnostic, not causal next-frame forecasting.

## Data integrity and coverage

The selection receipt reports 512 groups per category, 2,560 unique selected
sample IDs, 10,240 selected RGB clips, zero repeated RGB clip hashes, and zero
exact selected-clip overlap with either freeform train or validation. All five
categories are equally represented. No records were removed after scoring.
The publisher does not provide scene/procedural-generation IDs in these
records; exact hashes cannot rule out semantically related generated scenes.

All 100 pinned source TFRecord objects (685,580,352 bytes total) were checked
against the manifest. Local re-hashing verified all 100 sizes and SHA-256
receipts; all manifest MD5/size receipts matched. The extractor read and
validated the full source shards, including TFRecord framing CRCs and schema.
The selected compressed feature archive is `[2560,4,8,768]`, 234,249,318
bytes, SHA-256
`e18ecd5046d9a06920d998912a4c49b141aa50d511d876c95f192bf1e038f871`; its
contiguous feature values have SHA-256
`bc1339799004139d3426a04a59f6b746e5ed74fc89550bb8a5778372f50a7597`.

Freeform train/validation audit hashes were
`9b2bc4dc6138ab9839780f16570f621a87955dec611a28b6eb274b3f9a6635af` and
`c9db372663f6f8c89cb50c38537863d939d79195178fde12114592c3a85f34cd`;
the frozen split-manifest hash was
`03e7348c2144d5e2037d53692b8961a5f6376425dba23832b6f32122cd26e679`.

## Results

The primary metric is group-level ranking: did the model assign greater mean
surprise to the two impossible videos than to the two possible videos in the
same source quadruplet?

| Probe category | Groups | Group ranking accuracy | Clip accuracy | Clip ROC AUC | Clip NLL | Clip Brier | Clip ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuity | 512 | 0.0000 | 0.5000 | 0.4169 | 2.6337 | 0.4727 | 0.4717 |
| Solidity | 512 | 0.0918 | 0.4937 | 0.4680 | 0.9420 | 0.3337 | 0.2613 |
| Object Persistence | 512 | 0.7324 | 0.4932 | 0.4957 | 1.5652 | 0.3953 | 0.3574 |
| Unchangeableness | 512 | 0.1133 | 0.5000 | 0.4806 | 3.0933 | 0.4955 | 0.4952 |
| Directional Inertia | 512 | 1.0000 | 0.5820 | 0.8734 | 0.8308 | 0.2729 | 0.3202 |
| **Macro / pooled** | **2,560 / 10,240 clips** | **0.3875** | **0.5138** | **0.5469 macro / 0.5428 pooled** | **1.8130** | **0.3940** | **0.3812 macro / 0.3773 pooled** |

The stratified paired bootstrap over whole quadruplets used 10,000 resamples,
seed 17: macro group-ranking accuracy 0.3875, percentile 95% interval
[0.37695, 0.39805]. This interval is below chance for this fixed probe sample;
it describes this predeclared sample and does not remove any generalization
limits. Directional Inertia ranked perfectly while Continuity ranked in the
opposite direction, demonstrating strong category dependence.

At the fixed freeform-validation q95 threshold, pooled clip sensitivity was
0.9881, specificity 0.0395, accuracy 0.5138, and AUC 0.5428. Diagnostic
probabilities were derived from a one-class possible-only validation
distribution. Therefore NLL, Brier, and ECE are **not calibrated probability
quality claims**; the threshold was not fit on impossible clips. The
quadruplet ranking metric is the clearer result for this two-possible/two-
impossible task.

## Execution and immutable artifacts

The successful feature extraction ran on local CPU only, with
`CUDA_VISIBLE_DEVICES` empty and `gpu_compute_used=false`; no cloud compute or
GPU compute was used. It processed all 100 pinned source shards, 2,560 groups,
and 10,240 clips. Wall time was 11,936 seconds (3 h 18 m 56 s), mean forward
time 4.588 s/group or 1.147 s/clip, peak working set 4,210,745,344 bytes. The
feature-extraction report records Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1,
NumPy 1.26.4, Windows 10, and a 20-logical-CPU Intel processor.

The one-time evaluator ran on CPU under Python 3.11.9 / PyTorch 2.6.0+cu124
and completed in 3.495 seconds. It did not select or modify a model and records
`tuning_after_probe=false`.

Attempt history:

1. Attempt 01 downloaded and verified all source objects but stopped before
   feature extraction because a generic TFRecord audit imposed a too-small
   row-size limit on valid 984,684-byte quadruplet records. Payloads are kept
   for reuse.
2. Attempt 02 re-used and re-hashed the payload but was stopped before scoring
   after review found a deterministic read-only-array preprocessing failure;
   it produced no feature scores.
3. Attempt 03 used the pinned TensorFlow C++ sequential TFRecord reader,
   preserved CRC checks, used out-of-place preprocessing, passed the complete
   exact-media-overlap preflight, and completed feature extraction. These
   changes are recorded in the successful extraction receipt.
4. The first evaluator invocation correctly failed closed before loading probe
   features because the evaluator expected the frozen threshold hash in the
   extraction report, while the existing immutable selection receipt already
   carried it. The implementation now requires that selection-receipt binding
   and accepts a matching redundant report field when present; new extraction
   reports also write that field. No existing data or receipt was edited. The
   single scoring invocation then passed the unchanged threshold replay and
   receipt gates.

External artifacts remain under
`C:\CodexArtifacts\pretrained-reuse-r2-local\physical-concepts-probe-evaluation-20261127-attempt-03\`:

- `sample-selection.json` SHA-256
  `b556d8fe2cc7be8812cc346b03ed4e069d038afe96bc637be1cb86ec3496403e`
- `feature-extraction-report.json` SHA-256
  `429eafc01f4ddd8ec18a8c7c111d816459c9ce1b929a7bfe7478de4bc8e7dde2`
- `pre-probe-validation-threshold.json` SHA-256
  `4da87fd93645e1d691c5c28a69719d278d2854df916f5083e2a39d09fd1cb93c`
- `evaluation/evaluation-report.json` SHA-256
  `24a06a765b751fa45d419a1f76004827a10065e5fd596921a509c127a6db940a`
- `evaluation/quadruplet-predictions.jsonl` SHA-256
  `0b33348b81378e50220bd67529f9a97c9803b6478a5e8f645cad5a18523d2a83`
- `evaluation/clip-predictions.jsonl` SHA-256
  `d9755b09f1ccc58253c21750c935aa7f3ac9979bcef45c55102aef502550b164`

Prediction files preserve IDs, source generations, labels, scores, predictions,
fixed-threshold probabilities, and group outcomes. No media, features, model
weights, or prediction payloads are committed to Git. Existing Teacher,
IntPhys, and other legacy artifacts remain unchanged.

## Interpretation and next work

This result does not support reusing next-feature prediction error as a general
physical-reasoning decision score. Because the integrity gates passed, the
immediate bottleneck is better framed as an objective/domain mismatch than a
file-corruption or exact-duplicate problem. The large category spread warns
against treating one aggregate as evidence that every probe category behaves
similarly. The source still lacks scene/procedural IDs, and its probe camera
domain differs from freeform training.

Do not tune, recalibrate, resample, or retrain against these now-observed probe
results. Any next video Decision experiment needs a separately admitted,
task-matched training source with explicit decision labels and scene-family
grouping, followed by a newly frozen evaluation split. A task-matched BERT
experiment can be pursued independently for text only; it cannot answer this
video failure. The project-wide 95–98% four-modality objective remains unmet.
