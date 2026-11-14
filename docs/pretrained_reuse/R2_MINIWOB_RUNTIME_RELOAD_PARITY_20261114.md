# MiniWoB full-cohort runtime reload parity — 2026-11-14

## Scope

This is a mechanical save/reload parity check for the already observed
`click-color-v1:screen-only` prospective development cohort. It reloads the
existing pinned DINOv2-small encoder and selected epoch-99 readout through the
production-style `MiniWoBDinoDecisionRuntime`, then compares probabilities by
sample ID with the already saved matched-screen predictions. It does not
recompute or report task metrics, select a checkpoint, fit or calibrate a
model, or access the sealed audit. The cohort remains closed to tuning.

## Frozen inputs and integrity

- Source commit: `246686b523a1ffd775eb6607b0db5b10084fc027`.
- Frozen cohort: 1,024 rows; corpus manifest SHA-256
  `1a7b35e681677739bac456ff8a5fbba3cfbb4ba09da1e3d6ed6f501f4f2df84d`;
  row JSONL SHA-256
  `5b5203ef8a8afbbafe1e41ed66196ffa362e548ef7f4a6b0309408651496d87e`.
- Prior prediction JSONL SHA-256
  `ea8cd018f876a1bedd77531556658097b2ae3ff0b17c8b3fef7762d0b81988a3`.
- DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`,
  22,056,576 frozen parameters, weights SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`;
  processor SHA-256
  `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828`.
- Selected readout SHA-256
  `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`.
- The preflight compared the frozen experiment config SHA with the corpus
  manifest and verified all model and processor hashes before loading.

The first launch stopped at the corpus-status guard before model loading because
the verifier expected the wrong status string. Reading only the manifest header
showed the actual frozen status, `FROZEN_PROSPECTIVE_EVALUATION_NO_MODEL_SCORING`.
The guard was corrected to that exact value and an explicit config-hash match
was added; the replay then passed. No model/data files were changed.

## Result

The replay used Python 3.11.9 and PyTorch 2.6.0+cu124 on CPU (2 Torch threads).
The encoder remained frozen. All 1,024 sample IDs were replayed in their frozen
order, and every output was finite and normalized. The maximum and mean
absolute probability differences against the saved matched-screen predictions
were both **0.0** (fixed tolerance `2e-6`).

Runtime loading took 5.455 s. Replaying the 1,024 single-image requests took
86.541 s, or 84.513 ms/request in this one run. This is a CPU parity timing,
not a latency distribution or target-device performance claim. No GPU or cloud
compute was used.

The replay compared only IDs and probability vectors. Target fields in prior
evaluation artifacts were not used; no accuracy, NLL, Brier, ECE, or other
quality metric was recalculated. This adds runtime handoff evidence only and
does not broaden the synthetic screen-task result to natural images, video,
audio, text, or multimodal fusion.

## Reproduction artifacts

The external artifact directory is
`D:\CodexArtifacts\pretrained-reuse-r2\miniwob-runtime-reload-parity-20261114-attempt-01`.

- `replay.py` SHA-256:
  `ec174d31a42d57566b4c2ffa8a6111701f1bbc61e26026d840296f09bcf99d86`
- `result.json` SHA-256:
  `e97bb9aa08efcab05635085eccddf7a61dba6751ecf9d92e6d3e48d42e0ec428`
- `replay-probabilities.jsonl` SHA-256:
  `27b219282abf39790be7691b05daf1729b8f1caf91dcb259da7ef7d8b317d88a`

The code remains in the repository; corpus, checkpoints, and replay outputs
remain external artifacts.
