# CLEVRER temporal task-mix BiLSTM protocol (2026-10-18)

## Question

Does preserving the eight ordered VideoPrism frame-position features in a small
bidirectional LSTM help a frozen-feature Decision readout beyond the same
question-and-candidate text alone? This is a scoped video experiment, not a
claim about the complete Omni model or product quality.

## Frozen data

- Source: MIT-IBM/CLEVRER training questions at revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0).
- Seed: 17. The deterministic selector excludes previously observed scene
  identities, then groups the remaining records by underlying video scene.
- Training split: 1,024 rows from 614 scenes, with realized mix
  temporal descriptive 204, explanatory 306, predictive 306, and
  counterfactual 208 rows. (The source mix is intended to be 20/30/30/20; the
  recorded realized count is authoritative.)
- Validation split: 1,400 rows from 800 disjoint scenes; 200 scenes per task
  type. Reasoning parents contribute one correct and one wrong candidate row.
- The selector enforces scene, parent-question and normalized question/options
  separation. Media materialization verifies per-video SHA-256 and split
  identity. No sealed-audit data are loaded.
- Split and media manifest hashes are saved in the external preflight and run
  metadata; model weights and media remain outside Git.

## Input representation

For every reasoning candidate, the MiniLM input is the original CLEVRER question
plus the candidate statement. It is not inferred from the binary label. The
temporal descriptive examples retain their original multi-choice question and
taxonomy options. MiniLM-L3 and VideoPrism are frozen. VideoPrism supplies eight
ordered frame-position features; the candidate model is a 128-hidden-size
bidirectional LSTM followed by a shared candidate MLP. A text-only candidate MLP
is the matched reference. Both models score the same frozen train and validation
rows and return softmax probabilities over each row's supplied options.

The configured MiniLM-L3 hidden width is 384. Candidate text features
concatenate four 384-wide vectors, so their width is 1,536. The
VideoPrism+MiniLM+BiLSTM+candidate-readout total is 132,904,193 pre-quantization
parameters (VideoPrism 114,365,184; MiniLM 17,389,824; trainable temporal
readout 1,149,185), under the provisional 200M target. This accounting is for
this experiment's component only, not the full product model.

## Fixed fit and selection

- Seed 17; fresh readout initialization for each arm.
- AdamW, learning rate 0.001, weight decay 0.001.
- 64 epochs, batch size 32 questions, no performance early stopping; at most
  2,048 optimizer updates per arm.
- Each frozen training row is visited once per epoch, shuffled with seed 17.
- PyTorch deterministic algorithms and deterministic cuDNN mode are enabled;
  the GPU gate sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the child starts.
- Checkpoint selection is minimum validation macro question-type NLL; strict
  ties retain the earlier epoch. Final and selected checkpoints are saved
  separately, and the selected checkpoint is reloaded and compared exactly.
- Every epoch records train/validation Accuracy, NLL, Brier and ECE by task.
- A paired validation scene bootstrap compares the selected VideoPrism+BiLSTM
  arm with text-only, stratified by task type (10,000 replicates, seed 17).
  Because validation selects the checkpoints, these intervals are descriptive
  and are not independent-test guarantees.

## Compute and safety

Use only the local RTX 3080 Laptop; no cloud compute or purchases. GPU work is
launched behind a cooperative GPU-yield gate. It waits for a stable idle GPU,
then reserves 3.2 GiB so the reported 14,000 MiB free-memory gate for the
Orbitune supervisor remains closed while the child job uses the remaining
memory. It allows at least 8 GiB for the child and refuses to launch if those
limits cannot both be met. It does not stop or modify Orbitune. If the GPU does
not yield, the job does not launch. Sealed audit, existing Teacher artifacts
and prior runs remain read-only.

## Execution result

- Materialization completed for 1,024 training rows / 614 train videos and 1,400
  validation rows / 800 validation videos. Per-video hashes were recorded;
  scene, media-hash and normalized question/options overlap were all zero.
- Frozen-feature preflight passed with 1,414 unique media and no sealed audit.
  Train and validation ordered-ID hashes are in the external preflight.
- The VideoPrism extraction environment exposed `TFRT_CPU_0` only. It therefore
  extracted the fixed eight-frame features on CPU in 1,844.892 seconds. The
  resulting 32,421,494-byte feature cache SHA-256 is
  `a57b15480444dff2db80feed639c51d59e4e6bf0119bb33ff6af4ffcb8959a00`.
  The first run attempt held a GPU reservation unnecessarily; it did not use
  the GPU for feature computation. The readout fit remains CUDA-gated.
- The first CUDA readout attempt used the v1 output directory and stopped before
  any optimizer update: effective config requested 3,072 text features, while
  the four-vector MiniLM representation is 1,536 wide. Its directory is
  preserved with only `effective-config.yaml` and `preflight-used.json`; it is
  not a model-quality result. A preflight assertion now checks configured
  width against the pinned local MiniLM `config.json`. The corrected v2 config
  uses new, separate preflight and run directories and the same frozen data and
  feature cache.
- The v2 configuration preflight passed; config SHA-256 is
  `e80f66ee846e6baf233dfca826d5f77a4a1ef0e7f391c0812de9fedba3a7bc04`,
  preflight SHA-256 is
  `db8e81739dc572837133da50512a83121ea182d958c610a873e6e8d69e729f43`,
  and the source commit was `796a295bd61b07447df0ee73afe54dd13ad55e76`.
- One CUDA run completed both matched arms, each for 2,048 updates / 64 epochs
  and 65,536 train-row presentations. It used fresh readout and BiLSTM weights,
  not the failed v1 attempt; the frozen train and validation rows, order and
  encoders were unchanged. Run metadata SHA-256:
  `8efc42511b1fa5201b1e2b32fc7dfb5c0c27a145ec8b30a2c55a1606b50f31d0`.
  The external run directory contains effective config, preflight receipt,
  selected and final weights, per-epoch histories, per-example validation
  predictions and metadata. No model weights or media were added to Git.
- Selected checkpoints were much earlier than the final step: text-only epoch 5
  / update 160 (5,120 presentations), temporal epoch 7 / update 224 (7,168
  presentations). Selection used validation macro question-type NLL; final
  2,048-update checkpoints are retained separately.

### Validation result

Metrics below are macro averages over the four question types. The temporal arm
minus matched text-only differences were +0.0031 Accuracy, -0.0040 NLL,
-0.0017 Brier and -0.0063 ECE. All are small. The paired, scene-clustered
10,000-replicate 95% intervals include zero for Accuracy `[-0.0175, 0.0231]`,
NLL `[-0.00886, 0.00082]`, and Brier `[-0.00422, 0.00078]`; they are
development-validation diagnostics after checkpoint selection, not blind-test
confidence guarantees.

| Selected arm | Accuracy | NLL | Brier | ECE | Best update | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM text-only | 0.4575 | 0.8506 | 0.5482 | 0.0486 | 160 | 196,865 |
| VideoPrism 8-frame + BiLSTM + text | 0.4606 | 0.8466 | 0.5464 | 0.0423 | 224 | 1,149,185 |

| Question type | Text Accuracy / NLL / Brier / ECE | Temporal Accuracy / NLL / Brier / ECE |
|---|---|---|
| Temporal descriptive (n=200) | 0.3600 / 1.3215 / 0.6913 / 0.1248 | 0.3550 / 1.3071 / 0.6858 / 0.1287 |
| Explanatory (n=400) | 0.4900 / 0.6946 / 0.5014 / 0.0298 | 0.4825 / 0.6934 / 0.5003 / 0.0273 |
| Predictive (n=400) | 0.4825 / 0.6935 / 0.5003 / 0.0230 | 0.4950 / 0.6931 / 0.5000 / 0.0076 |
| Counterfactual (n=400) | 0.4975 / 0.6928 / 0.4997 / 0.0170 | 0.5100 / 0.6928 / 0.4996 / 0.0057 |

The fixed learning-curve points show increasing overfit, not a delayed gain.
| Arm | Update | Train batch CE | Validation macro Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Text | 512 | 0.6978 | 0.4594 | 0.9430 | 0.5898 | 0.1148 |
| Text | 1,024 | 0.5537 | 0.4537 | 1.2158 | 0.6593 | 0.2045 |
| Text | 1,536 | 0.4421 | 0.4381 | 1.5675 | 0.7344 | 0.2779 |
| Text | 1,920 | 0.3678 | 0.4594 | 1.7941 | 0.7616 | 0.2954 |
| Text | 2,048 | 0.3428 | 0.4513 | 1.8637 | 0.7807 | 0.3256 |
| Temporal | 512 | 0.7185 | 0.4631 | 0.9093 | 0.5771 | 0.1074 |
| Temporal | 1,024 | 0.5246 | 0.4419 | 1.1803 | 0.6579 | 0.2112 |
| Temporal | 1,536 | 0.2604 | 0.4431 | 1.6143 | 0.7819 | 0.3295 |
| Temporal | 1,920 | 0.1073 | 0.4488 | 2.0123 | 0.8498 | 0.3708 |
| Temporal | 2,048 | 0.0721 | 0.4494 | 2.0885 | 0.8594 | 0.3847 |

This supports severe overfitting under the fixed training recipe. The small
selected-checkpoint difference does not establish that temporal features help;
the bootstrap intervals are wide enough to include no difference. This is a
scoped CLEVRER probe, not evidence of general video quality or multimodal
product readiness. No sealed audit was loaded.

### Hardware and handoff caveat

The run used the local RTX 3080 Laptop GPU, PyTorch 2.6.0+cu124, CUDA 12.4,
Python 3.11.9, and Transformers 5.6.2. Recorded child peak allocated VRAM was
153,878,016 bytes (peak reserved 209,715,200 bytes); both fits together took
128.2 s training and 28.7 s evaluation. Cloud cost was $0. VideoPrism frame
feature extraction remained CPU-only as noted above.

The first HotUGPU wrapper had a handoff race: after Orbitune cleanly yielded,
its supervisor relaunched during the CUDA job's startup. Its log records the
second yield request at 21:39:51 JST and clean trainer exit at 21:40:07; this
fit ended at 21:40:14. The run completed without OOM or non-finite values, but
the card was not exclusively allocated for the entire startup window. The
wrapper is now changed to create a tiny CUDA handoff guard before waiting, so
the supervisor sees the foreign CUDA PID while its trainer exits; a focused CPU
test covers the ordering. This correction is for future launches and did not
alter the completed run.

No extra seeds, hyperparameter changes, audit evaluation, model promotion,
quantization, or other modalities were attempted.
