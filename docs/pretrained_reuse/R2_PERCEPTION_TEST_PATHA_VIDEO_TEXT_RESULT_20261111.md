# R2 Perception Test Video + Text Path A — 2026-11-11

## Result and scope

The frozen Path A reader completed its planned one-time comparison on the
public validation candidate pool. It is a **development-candidate,
video-plus-text, three-choice MCQA** result, not a blind test, not evidence of
participant/script-independent generalization, and not a general Omni quality
result. The official Perception Test test split and all sealed audits remained
unopened. No tuning followed this evaluation.

The joint reader was slightly more accurate than text-only, but its Accuracy
difference is uncertain under the predeclared video-group bootstrap, while its
NLL and ECE are worse. Correct video versus a deterministic different-video
control improves NLL and Brier, suggesting the video features carry some useful
signal for this reader. These results do **not** establish a reliable overall
joint-modality benefit or product readiness.

## Frozen inputs and integrity

- Source: `google-deepmind/perception_test`, revision
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; source metadata declares CC
  BY 4.0.
- Corpus manifest: `D:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-patha-v0-attempt-01\manifest.json`.
- Train: 7,392 questions / 1,955 video groups;
  record SHA-256 `2f2fa8ed4e81baaadcede50cc4343f0f0c55a02dd63703d3313e8d2188cb60a7`.
- Public validation candidate: 2,886 questions / 1,518 video groups;
  record SHA-256 `2090991bf3a956c67e31cb3d3b6cf68e60fbaf29fdf90aae640a110f3c674350`;
  ID order SHA-256
  `6c3c72ac4e0a304f6d6a61726e754155d9153c877d001c9390d96acd9bb191b4`.
- Video-group train/validation overlap: 0. Exact normalized question/choice
  overlap: 0. All 1,518 validation video feature shards matched their receipts;
  each was verified for SHA-256, `[8,14,14,384]` shape, float32 dtype, and
  finite values before evaluation.
- Validation reasoning counts: descriptive 2,281; explanatory 295;
  predictive 93; counterfactual 217. The mix is heavily descriptive.
- Validation source/media exposure is **not proven absent**. The fixed pool was
  already described as a public-validation candidate; no participant/script
  identifiers establish independence. Treat these measurements as exposed
  development evidence. No official test or sealed audit data was opened.

## Model and training record

- Video encoder: OpenGVLab VideoMAE V2 Small, 22,153,286 parameters;
  checkpoint SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`;
  Hub revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`.
- Text encoder: MiniLM-L3, 17,389,824 parameters; model weight SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`;
  revision `4ca70771034acceecb2e72475f72050fcdde4ddc`.
- Reader: four-head query-to-video cross-attention and shared candidate scorer;
  444,673 trainable parameters. Encoders were frozen. The resident parameter
  subtotal for this video-text path is 39,987,783; it is not a unified
  four-modality package measurement.
- Seed 17; AdamW, learning rate `3e-4`, weight decay `1e-4`, batch 16, maximum
  30 epochs, patience 6; lowest internal-selection NLL, Brier tie-break within
  `1e-5`. The run completed 8 epochs and selected epoch 2; epoch 8 is retained
  separately as the final checkpoint.
- Internal selection used 1,111 questions / 294 video groups. Training used
  6,281 unique questions / 1,661 groups. Total training passes across the full
  run: 50,248 questions; selected epoch: 12,562 passes over 6,281 unique train
  questions. Public validation was not used for fitting or checkpoint
  selection.
- Overfit signal: training NLL fell from about 0.737 at epoch 1 to 0.193 at
  epoch 8, while internal-selection NLL rose from 0.752 at selected epoch 2 to
  1.234 at epoch 8; selection ECE rose from 0.0457 to 0.2007. Best and final
  checkpoints remain distinct and preserved.
- Effective training config SHA-256
  `9d63d29fec6b56a5281ad7ec9d7ec6919b87b31a7e2fe430da7b00c2e73baa0a`.
  Best checkpoint SHA-256
  `e9516be6d58ba82c3c4f111e59d6b724e856b3ac412bbc2c57b683d35deecb8c`;
  final checkpoint SHA-256
  `ac499410701563464b65126a0558ef71e45d965410d17410e35f249a348832ab`.

## One-time validation metrics

All arms use the same 2,886 examples and targets. ECE is 15-bin.

| Arm | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Correct video + text (joint) | 0.5177 | 1.0026 | 0.5905 | 0.0837 |
| Text-only (zero video memory) | 0.5083 | 0.9847 | 0.5883 | 0.0319 |
| Text + deterministic mismatched video | 0.5059 | 1.0194 | 0.6024 | 0.0720 |
| Uniform three-choice reference | 0.3333 expected | 1.0986 | 0.6667 | 0.0000 |

### Joint reader by question type

| Type | Count | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Descriptive | 2,281 | 0.5300 | 0.9712 | 0.5726 | 0.0748 |
| Explanatory | 295 | 0.4847 | 1.1809 | 0.6752 | 0.1574 |
| Predictive | 93 | 0.4409 | 1.1286 | 0.6747 | 0.1634 |
| Counterfactual | 217 | 0.4654 | 1.0367 | 0.6269 | 0.1187 |

### Paired video-group bootstrap

2,000 percentile bootstrap replicates, seed 17, resampling unit = video ID;
all questions for each selected video stay together. Delta is joint minus
control.

| Control | Metric | Delta | 95% interval |
|---|---|---:|---:|
| Text-only | Accuracy | +0.00944 | [-0.00485, +0.02396] |
| Text-only | NLL | +0.01795 (worse) | [+0.00128, +0.03468] |
| Text-only | Brier | +0.00209 (worse) | [-0.00642, +0.01091] |
| Text-only | ECE | +0.04897 (worse) | [+0.02543, +0.07066] |
| Mismatched video | Accuracy | +0.01156 | [-0.00105, +0.02478] |
| Mismatched video | NLL | -0.01652 (better) | [-0.03320, -0.00186] |
| Mismatched video | Brier | -0.01187 (better) | [-0.02016, -0.00426] |
| Mismatched video | ECE | +0.01163 (worse) | [-0.00213, +0.02523] |

The fixed option permutation `[2,0,1]`, restored to original option order,
changed no logits (`max absolute delta = 0`) and caused no prediction flips.
This verifies candidate-order equivariance for this checkpoint and inference
path; it does not address accuracy or generalization.

## Runtime and artifacts

- Hardware: local NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB; PyTorch
  `2.6.0+cu124`; CUDA runtime 12.4; Python 3.11.9. No cloud compute or spend.
- Recorded GPU time: train video-feature extraction 6,089.55 s; validation
  video-feature extraction 4,833.96 s; reader training 344.79 s; one-shot
  evaluation 27.53 s. Combined recorded stages: 11,295.83 s (about 3 h 8 min
  16 s). Validation evaluation used 437,037,568 bytes peak PyTorch allocation
  and took 27.53 s.
- Validation VideoMAE feature manifest SHA-256
  `8a4e8c01b8752188efeb75ee2c91a4144860ddb00d0e8998da52c52485087fbb`;
  validation MiniLM feature file SHA-256
  `53f044310060ebfad788832afc9503372295128f7518eb8a8f9600f8cad7a76f`.
- Evaluation report:
  `D:\CodexArtifacts\pretrained-reuse-r2\perception-test-video-readout-eval-attempt-01\evaluation-report.json`
  SHA-256 `d3e5b1084c3109afe83b9d3f9fee21d386bb705aaac1e49231d67b88eeecd5d2`.
- Per-example predictions:
  `D:\CodexArtifacts\pretrained-reuse-r2\perception-test-video-readout-eval-attempt-01\validation-predictions.jsonl`
  SHA-256 `efa74a3596164aa343ce1ddc4e350987f178b25328c634eeb1b7f283f3304212`.
- One-shot lock:
  `D:\CodexArtifacts\pretrained-reuse-r2\perception-test-video-readout-attempt-01\public-validation-evaluation-lock.json`.
  Its final state is `COMPLETE`; it records evaluation report and prediction
  hashes. The original run metadata still says `RUNNING`, although the process
  is terminal and the validation feature manifest says `COMPLETE`; treat the
  feature manifest and process observation as authoritative. This stale
  `run.json` status is recorded and was not retroactively edited.
- Worktree source commit: `6e679ee8c1a5aeda8280a56030099b2ed78cf2ea`;
  evaluation script SHA-256
  `088fb7d007064c64d49478a1c25dad3c4a87621ca3f83351c77c810314ef0212`.

## Decision and remaining gates

This one run does not justify tuning on these validation labels or promoting
the reader. The paired Accuracy intervals include zero; joint calibration is
materially worse than text-only; small reasoning subsets have limited
precision; independent script/participant exposure is unknown. Correct video
beats mismatched video on NLL/Brier, but that does not establish general
multimodal capability. Do not infer a VideoMAE backbone ceiling from this
small, overfit reader.

The experiment is a measured R2 Path A implementation and one-time comparison,
not completion of R0–R8. It handles only a single video plus question/options;
image/audio inputs, synchronized cross-modal necessity, a production-calibrated
JSON runtime, full package size/latency, and the project quality gates remain
open. Existing Teacher v1, other v2 artifacts, media, and checkpoints were not
modified; no model/media/checkpoint was added to Git.
