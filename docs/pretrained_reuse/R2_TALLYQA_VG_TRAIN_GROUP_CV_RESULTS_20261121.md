# TallyQA-VG TRAIN-only grouped CV results (2026-11-21)

## Outcome

The fixed five-fold, nested group CV completed over all 3,101 source TRAIN
questions. The selected frozen VideoPrism-B + MiniLM joint readout reached
55.37% OOF accuracy, 1.0711 NLL, 0.5820 Brier, and 0.0420 ECE-15. The
question-only control reached 56.56%, 1.0753, 0.5821, and 0.0418. The joint
model is marginally better on NLL/Brier but worse on accuracy; all paired
group-bootstrap intervals include zero. This does not establish an OOF
improvement from adding image features.

The post-fit zero-image control fell to 53.37% accuracy, 1.1476 NLL, 0.6227
Brier, and 0.0951 ECE-15. On this fitted readout, image features affect output,
but that sensitivity does not show that they improve held-out performance.

This is retrospective development evidence on a project-created four-choice
adaptation of an open-response counting dataset. Existing task and model
choices had been informed by an exposed development validation set. The
result is not an independent product-quality estimate, is not comparable to
the original TallyQA open-response metric, and does not close Path A, R2-M, or
product-admission gates.

## Frozen protocol and inputs

- Source commit: `11e49cfb9314312bb0224c8be20cb81365155dbc`.
- Protocol: [TRAIN-only grouped-CV protocol](R2_TALLYQA_VG_TRAIN_GROUP_CV_PROTOCOL_20261121.md).
- Seed 17; five outer folds; inner five-fold connected-group assignment with
  fold 0 used only for NLL checkpoint selection; earliest minimum NLL selected.
- Connected components join rows sharing `image_id` or normalized question
  text, transitively. Outer fold assignment is deterministic, label-blind,
  largest-component-first, then least-loaded. The saved assignment hash is
  `4fd97107a389dcc1dce46057043a0bb25b6ab5cf687a4744c97c0d5964a007b8`.
- Source task manifest SHA-256:
  `59c1c03edf17a8da6c870d4827d033af58759c8300f953b60cb0abbff98f8a45`.
- Train-only manifest SHA-256:
  `40857d61ebec3377c74ab037cb0dd897dcfe1026a0a6558d89c42a1c55da3cde`.
- Frozen image cache SHA-256:
  `ddd738a362d7b31762c676dc178fed0183ddce8052ffaac4bbc93f57ef2753a6`.
- Cache report SHA-256:
  `aa018cfcac4e714d26d43d7c7810193555dee7ebc02d848a6fa619bfe1e1df6d`.
- Effective readout config SHA-256 (recorded in `result.json` and recomputed
  from the checked-out file):
  `a391344912206d29ff432d9ded7e6dc347627312f378a20f92971736af7cbc9d`.
- VideoPrism-B checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
  114,365,184 frozen parameters.
- MiniLM-L3 revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, weights
  SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`;
  17,389,824 frozen parameters.
- Joint readout: 196,865 trainable parameters; total pre-quantization count
  131,951,873. Question-only readout: 98,561 trainable parameters.
- Inputs, source code hashes, software versions and per-fold histories are
  recorded in the external `run/result.json`; selected checkpoints and OOF
  predictions are also retained there. The original run directory is
  `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\tallyqa-vg-train-groupcv-v1-attempt-03\run`.

The original terminal command line was not persisted in run metadata, so it is
`UNKNOWN`. The run is identified by the recorded source commit, effective
config and input hashes, preflight receipt, source-code hashes, environment,
and result file. Its equivalent required invocation is:

```powershell
C:\CodexArtifacts\venvs\embeddinggemma2-v518\Scripts\python.exe -m scripts.cv_tallyqa_vg_train_groups `
  --config configs/pretrained_reuse/path_a_tallyqa_vg_videoprism_seed17.yaml `
  --task-manifest <attempt-03>/train-only-task-manifest.json `
  --feature-cache <frozen-source>/feature-cache/features.npz `
  --cache-report <frozen-source>/feature-cache/cache-report.json `
  --text-model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --output-dir <attempt-03>/run `
  --preflight-receipt <attempt-03>/preflight.json `
  --expected-preflight-sha256 8ba840d595bf8970377e9a6d483718f9cc83c2109535e4c627643d83703e8def
```

The runner's timer covers the outer fold fitting/scoring loop and group
bootstrap, not initial feature loading/encoding. It recorded 47.95 seconds for
that timed section; artifact timestamps span 17:45:02–17:45:47 local time on
2026-11-21. The run used CUDA, PyTorch 2.6.0+cu124, NumPy 2.4.6, and
Transformers 5.19.0. No run-specific peak VRAM or complete start/end timestamp
was instrumented. No cloud compute or external cost was used.

## Results

| OOF arm | N | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Image + question | 3,101 | 55.37% | 1.0711 | 0.5820 | 0.0420 |
| Question only | 3,101 | 56.56% | 1.0753 | 0.5821 | 0.0418 |
| Joint checkpoint, image zeroed | 3,101 | 53.37% | 1.1476 | 0.6227 | 0.0951 |

Paired component-bootstrap point differences (joint minus question-only):
accuracy −1.19 percentage points; NLL −0.00426; Brier −0.000096. The frozen
5,000-replicate, seed-17 percentile intervals over transitive image/question
components were accuracy [−3.35, +0.61] pp, NLL [−0.0239, +0.0172], and Brier
[−0.0113, +0.0118]. Each interval includes zero. These are descriptive
intervals on the same TRAIN groups used for this nested-CV procedure, not a
new blind benchmark.

| Outer fold | Rows | Joint accuracy / NLL / Brier / ECE | Question-only accuracy / NLL / Brier / ECE |
|---:|---:|---|---|
| 0 | 621 | 60.06% / 1.0424 / 0.5553 / 0.0500 | 58.62% / 1.0489 / 0.5620 / 0.0887 |
| 1 | 620 | 50.81% / 1.1072 / 0.6025 / 0.0538 | 56.77% / 1.0987 / 0.5968 / 0.0786 |
| 2 | 620 | 55.65% / 1.0515 / 0.5673 / 0.0411 | 57.26% / 1.0625 / 0.5741 / 0.0550 |
| 3 | 620 | 54.52% / 1.1274 / 0.6129 / 0.0952 | 52.74% / 1.1648 / 0.6300 / 0.1055 |
| 4 | 620 | 55.81% / 1.0269 / 0.5722 / 0.0568 | 57.42% / 1.0019 / 0.5479 / 0.0550 |

The zero-image control's saved aggregate shows a 2.00 pp accuracy decrease and
proper-score degradation relative to the joint arm. It diagnoses input
sensitivity only; it is not an image-feature ablation retrain.

## Preflight and integrity

Two preflight-only attempts were preserved and did not fit or score a model:
attempt 01 incorrectly treated multiple questions sharing an image as a
duplicate-image error; attempt 02's label-stratified assignment yielded two
outer folds with only 18 rows. The label-blind component-size assignment was
frozen before attempt 03. Its 3,101 questions, 2,795 unique images, and 2,334
connected components were assigned to balanced outer folds of 621, 620, 620,
620, and 620 rows. Nested fit/stop/score components were disjoint.

After fitting, a read-only verifier checked all 13 indexed files (9,431,923
bytes), their exact SHA-256 and sizes, all 3,101 unique TRAIN question IDs and
their order, fold/target/image identity, four-way probability validity and
option binding, and recomputed OOF metrics. It then loaded the saved selected
checkpoints and replayed each exact outer TRAIN fold without fitting. The
maximum absolute probability difference was 0 for joint, question-only, and
image-zero arms; argmax flips were 0. The aggregate metrics agree with
`result.json` within `1.3e-8`.

Verifier report (written outside the repository):
`C:\CodexArtifacts\pretrained-reuse-r2\tallyqa-vg-train-groupcv-v1-attempt-03\checkpoint-replay-verification-02.json`.
Its SHA-256 is
`1fbe178b6fe93938d7ddd9ff0d194ab78f8cb28d89f64ff18f023929b8d68fc8`.
The original source run, its selected weights, and its OOF file were not
modified. Official validation, official test, and sealed audit were not
loaded or reopened.

## Decision and limits

The CV does not support the claim that adding image features improves
generalization for this task adaptation: the joint-versus-question-only
accuracy difference is negative and its paired interval includes zero, while
the small proper-score differences are also uncertain. The post-fit zero-image
control indicates the fitted readout uses visual features, but the effect is
not sufficient to establish beneficial image transfer.

Do not continue tuning this source based on these outcomes. Keep product and
Path A claims open. Next work should target a distinct, unused TRAIN decision
source or a separately admitted task, retain grouped leakage controls, and
measure complementary input use before merging experiments. Existing exposed
development validation, official test, and sealed audit are not part of this
result.
