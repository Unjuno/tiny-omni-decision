# FSD50K audio annotation-Decision preflight — 2026-11-15

## Task scope

This candidate predicts whether FSD50K assigned each of four supplied broad
labels: `Animal`, `Human_voice`, `Vehicle`, and `Music`. Each request is a
binary Decision with options `no_annotation` and `annotated`, serialized as
option probabilities. **This predicts the dataset's annotation convention. A
missing dev annotation is not evidence that the corresponding sound is
physically absent.** No broad audio-understanding claim follows from this task.

## Split and exposure gate

- Source: FSD50K, DOI `10.5281/zenodo.4060432`.
- Source archives were SHA-256 checked against the prior receipt:
  - document: `a5c314ad2bf93d01b5921aa4270c70c90e28ac9a283ba9f08167490e728deb85`
  - metadata: `9a738e032546f9a2c6e3d04566928d04a65fb79b422cc9d78bb781723537bd19`
  - ground truth: `db2396260a7b1fb06feb6ccac71685794fc89917d71c3a8d64a1de85c2cc0ebf`
- Only official `dev.csv` rows marked `train` enter the development cohort.
  Clips are retained only under their own CC BY 3.0 or CC0 license; NC and
  Sampling+ clips are excluded. The per-clip creator, title, source URL and
  license are preserved in every manifest row.
- All 129 previously selected FSD clip IDs and all 128 corresponding uploader
  groups were quarantined. This covers the prior 128-clip feature/training
  experiments and the two small acquisition smokes.
- Uploader groups are assigned deterministically from
  `SHA256("tiny-omni-fsd50k-annotation-v1:17:" + uploader)`: buckets 0–6 train,
  7 calibration, 8–9 validation. All eligible source rows are kept; no
  outcome-based sampling or class balancing is applied.
- Split counts:

| Split | Clips | Uploader groups | Animal labels | Human_voice labels | Vehicle labels | Music labels |
|---|---:|---:|---:|---:|---:|---:|
| Train | 12,845 | 2,089 | 1,213 | 1,088 | 977 | 2,667 |
| Calibration | 2,053 | 290 | 208 | 285 | 119 | 254 |
| Validation | 3,981 | 572 | 275 | 623 | 243 | 558 |

  The validation task exceeds the provisional 1,000-example/200-group gate.
  Positive counts are label counts, not verified acoustic detections.
- Train/calibration/validation have zero sample-ID or uploader overlap by
  construction and validation in code. Exact decoded-media duplicate checks
  remain a pre-fit gate.

## Official evaluation separation

The official FSD50K `eval.csv` labels and eval audio were not read or acquired.
During a separate metadata-only viability check, `eval_clips_info_FSD50K.json`
was read for license and uploader structure: 10,231 records, of which 8,403
are CC BY/CC0 across 1,932 uploader groups. Therefore the eval outcome labels
remain unopened, but its metadata and sample/group counts are no longer
unobserved. Do not use eval labels for fitting, calibration, candidate choice,
or repeated evaluation. Any later evaluation must follow a frozen candidate
and be reported as outcome-blind, not metadata-blind.

## Frozen feature route and current state

Candidate encoder is the existing R1-approved EfficientAT DyMN-0.4 checkpoint
(`fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806`,
1,961,239 parameters, checkpoint SHA-256
`49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`). The
preflight proposes whole-clip mono audio at native 44.1 kHz, resampled to
32 kHz, with the already-audited EfficientAT preprocessing and frozen 384-D
features. A `NativeFeatureDecisionScorer(384, 6, hidden_width=64)` can encode
the four-way label ID and the `no_annotation`/`annotated` response as fixed
typed fields; the 25,089-parameter readout remains separate from the frozen
encoder. The selected candidate must report their total parameter count.

No media was downloaded, no features/model were loaded, no training/evaluation
was run, and no GPU was used. The manifest was generated at
`C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-annotation-presence-v1-preflight-20261115-attempt-01/`.
Its canonical manifest hash is
`0bbe5db5e6167a4c041e349889920f8da0f1a0703ee00fa63ff12a59a0531b6d`; ordered
sample-ID hashes are recorded in its `preflight.json`. Official evaluation
labels remain unopened.

## Checks and next gate

`tests/test_fsd50k_annotation_decision_manifest.py`: 5 passed; Ruff passed for
the builder and tests. Before fitting, acquire only the frozen development
manifest, verify archive-member CRC and decoded-media hashes, and check exact
cross-split/media duplicates. Then extract frozen features, fit the one
predeclared typed Decision readout, fit calibration only on the uploader-held
calibration split, and select by validation-only macro NLL. The official eval
labels remain reserved until that candidate is frozen.

This is one audio annotation-prediction component of Path A. It does not close
R2-M joint-input necessity, text/image/video quality, or the overall product
gate.

## Independent local freeze revalidation (2026-11-15)

Re-read the frozen train/calibration/validation JSONL and independently recomputed the canonical ordered sample-ID and full ordered-row SHA-256 values against `preflight.json`. Counts and both hashes matched for all three splits (12,845 / 2,053 / 3,981). A pairwise audit found zero sample-ID, source-ID, or uploader-group overlap. This read-only check used only the frozen development manifests; no media, official eval labels, features, or model were loaded.

## Train-prevalence validation baseline and class-imbalance guard

Before fitting the readout, a non-fitted baseline was computed by predicting each label's train prevalence as the `annotated` probability on the frozen validation split. Because every train prevalence is below 0.5, this is equivalent to always selecting `no_annotation`. It obtains macro Accuracy **0.893306** (per-label: Animal 0.930922, Human_voice 0.843507, Vehicle 0.938960, Music 0.859834), macro NLL **0.342078**, macro Brier **0.096179**, and macro 15-bin ECE **0.044908**. This shows why Accuracy alone cannot establish useful audio decisions for this imbalanced task. Candidate reporting now also includes per-label positive precision/recall, balanced accuracy, and positive support; the predeclared validation macro-NLL selector is unchanged. The baseline is descriptive, not a fitted or selected model.
