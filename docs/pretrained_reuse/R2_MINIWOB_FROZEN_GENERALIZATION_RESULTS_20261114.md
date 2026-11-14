# MiniWoB frozen-checkpoint generalization result — 2026-11-14

## Decision and scope

This is a one-task, inference-only generalization check for the already selected
DINOv2-small screen-action readout. It asks whether the fixed scorer uses the
rendered screen to choose one of four click actions on fresh MiniWoB
`click-color-v1` simulator seeds. It does not test natural-image understanding,
video, audio, text-only decisions, cross-modal fusion, temporal memory, or
product readiness. The simulator task template is shared with the training
corpus; the claim is seed-level generalization inside this fixed synthetic task.

The frozen checkpoint was not trained, calibrated, or selected on this cohort.
The 512-row epoch-selection validation set was not loaded or rescored. No
sealed audit was loaded. This prospective cohort is now observed and must not be
used to tune the checkpoint, preprocessing, temperature, or task definition.

## Frozen source, cohort, and model

- Source: MiniWoB++ commit `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`, MIT;
  license SHA-256 `24c56d96c8a2e4e20c20867b14d177b801027b603c45622218b34c841ad4972a`.
- Cohort: first 1,024 eligible episodes in seed window
  `[1,008,192, 1,016,384)`. The collector attempted 1,805 seeds. Each accepted
  episode has a unique seed, episode group, and screenshot.
- Prior-use gate: 12,329 historical seed values and 2,145 decoded screenshot
  pixel identities were inventoried before the finalized cohort; train,
  validation, and both earlier smoke roots were included. Seed overlap and
  decoded-pixel overlap are both zero.
- Target-color labels are collector-only audit fields. The model receives only
  screenshot RGB, opaque candidate IDs, and candidate coordinates. The
  four-choice order is deterministic per example and then independently
  permuted for the order-invariance control.
- Encoder: DINOv2-small revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, 22,056,576 parameters, frozen;
  weight SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- Readout: previously selected epoch-99 checkpoint, 76,293 parameters,
  305,644 bytes; SHA-256
  `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`.
- Resident total: 22,132,869 parameters, of which zero encoder parameters are
  trainable. Temperature is fixed at 1.0. No model package size or quantized
  artifact was produced.

## Results

Metrics use all 1,024 examples. ECE is 15-bin top-label ECE. The matched-screen
95% intervals are 10,000 percentile-bootstrap draws over independent simulator
episodes (seed 17). Paired control intervals use the same episode resamples
(10,000 draws, seed 1701).

| Input condition | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Matched screen | 96.97% | 0.1223 | 0.0594 | 0.0511 | 0.9212 |
| Sattolo-shuffled screen | 24.90% | 5.2159 | 1.3725 | 0.6722 | 0.9212 |
| Black screen | 25.88% | 2.3441 | 0.9319 | 0.3360 | 0.5948 |
| Train-only candidate-position prior | 24.02% | 1.3881 | 0.7509 | 0.0273 | 0.2676 |
| Matched screen, candidate order permuted and remapped | 96.97% | 0.1223 | 0.0594 | 0.0511 | 0.9212 |

Matched-screen accuracy's episode-bootstrap 95% interval is **95.90–97.95%**;
NLL interval is 0.1051–0.1412, Brier interval 0.0482–0.0717, and ECE interval
0.0433–0.0616.

Compared with the Sattolo-shuffled screen, the paired matched-minus-control
Accuracy difference is 72.07 percentage points (95% interval 69.14–74.90
points). Its paired NLL difference is -5.094 (interval -5.365 to -4.825), and
paired Brier difference is -1.313 (interval -1.365 to -1.261). This supports
the narrow conclusion that the scorer depends on the correctly paired screen
for this task. Under a shuffled screen it remains highly confident and wrong,
so the control also exposes a serious out-of-distribution calibration weakness.

Candidate-order equivariance passed: after mapping probabilities back to
opaque option IDs, maximum absolute difference was `1.19e-7`, below the frozen
`2e-6` tolerance; there were no Accuracy changes. The train-only position prior
stayed near the four-choice chance rate.

### Accuracy by target color

These are descriptive strata, not separately powered benchmarks. Each color
has 68–90 examples.

| Target color | n | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Black | 74 | 100.00% | 0.0224 | 0.0068 | 0.0200 |
| Blue | 79 | 100.00% | 0.0189 | 0.0046 | 0.0171 |
| Cyan | 86 | 94.19% | 0.1523 | 0.0897 | 0.0601 |
| Grey | 82 | 100.00% | 0.0670 | 0.0188 | 0.0607 |
| Lime | 82 | 100.00% | 0.0783 | 0.0266 | 0.0689 |
| Magenta | 78 | 96.15% | 0.0824 | 0.0371 | 0.0620 |
| Olive | 68 | 95.59% | 0.1891 | 0.0975 | 0.0840 |
| Orange | 79 | 94.94% | 0.2215 | 0.1115 | 0.0687 |
| Pink | 80 | 95.00% | 0.1337 | 0.0661 | 0.0669 |
| Purple | 69 | 97.10% | 0.1304 | 0.0661 | 0.0582 |
| Red | 90 | 100.00% | 0.0612 | 0.0187 | 0.0549 |
| White | 74 | 91.89% | 0.2303 | 0.1350 | 0.0671 |
| Yellow | 83 | 95.18% | 0.2172 | 0.1045 | 0.0869 |

## Execution and recovery record

The first prospective collection attempt used `[1,000,000, 1,008,192)` and
created 1,024 screenshots, then failed before writing rows or a manifest
because the collector-side validator unnecessarily imported the optional Torch
runtime. That incomplete output was retained, its screenshot receipts were
hashed, and its full attempted seed window was excluded from the next run.

The attempt-02 collection produced all 1,024 screenshots but initially hit a
second serialization import-path error. A recovery command deterministically
replayed the same seed window and verified every preserved RGB screenshot
before writing rows. It did not overwrite or score the images. The finalized
manifest records zero historical seed and pixel overlap.

The first inference pass completed all four scenarios but failed while joining
collector-only color labels by an audit key that does not exist. No metrics or
predictions were persisted or inspected. After adding a tested seed-based join,
the same frozen model, cohort, preprocessing, and controls were run again and
fully persisted. No tuning decision was made from the failed pass. The final
evaluation must be treated as observed and closed to future development.

The successful inference ran on CPU: Python 3.11.9, PyTorch 2.6.0+cpu, NumPy
1.26.4, Pillow 13.0.0; CUDA was not initialized. Total evaluation wall time was
531.2 seconds and process CPU time was 799.0 seconds. Cloud and GPU compute cost
were both $0. The RTX 3080 Laptop GPU remained occupied by an unrelated
Orbitune run and was not used or interrupted.

Corpus collection used the already installed MiniWoB 1.1.0, Gymnasium 1.4.0,
Selenium 4.52.0, NumPy 2.4.6, and Pillow 13.0.0 environment without upgrades,
with Edge 157.0.4341.46 and a Microsoft-signed EdgeDriver
(`6c3fdcb995cd406228d2a7877695c6fd60bc81f35b678cf1e3b79de3ab58bd8f`).

## Verification

- Focused CPU unittest coverage: 6 passed, including fresh seed/pixel rejection,
  metric/bootstrap determinism, audit-label joining, and screenshot-preserving
  replay verification.
- Full repository CPU suite: `2904 passed, 11 skipped` with CUDA hidden.
- `ruff check .`: passed.
- Training and evaluation dataset catalog validation: passed (9 and 11 sources).
- Python compilation of the collector, recovery, evaluator, and metric modules:
  passed.

## Reproduction artifacts and hashes

Local immutable run root:
`D:\CodexArtifacts\pretrained-reuse-r2\miniwob-frozen-generalization-20261114-attempt-02`

| Artifact | SHA-256 |
|---|---|
| Effective config | `f85740038323344acb311a62bc42982f08f751a9e4285a727f237878f93f990e` |
| Corpus manifest | `1a7b35e681677739bac456ff8a5fbba3cfbb4ba09da1e3d6ed6f501f4f2df84d` |
| Evaluation rows | `5b5203ef8a8afbbafe1e41ed66196ffa362e548ef7f4a6b0309408651496d87e` |
| Collector audit | `6c1b9f5d146a40a54386c763513df24f8e665f4f3b05b22f9c64139e51132fde` |
| Prediction records | `ea8cd018f876a1bedd77531556658097b2ae3ff0b17c8b3fef7762d0b81988a3` |
| Results | `c4eef34de653ebe382757523834ee9109513875f0e243c24741e9fa7d709304d` |
| Original run metadata | `ce9b0c2125947f633ce0677960c5a31081255818f401735094377c4b8b74b294` |
| Provenance audit | `7518e1fc8e0e1e9d37d41efa8f86e01b78582be99e1ed125d6d8109682b1d8ee` (`evaluation/provenance-audit.json`) |

The execution's Git base was `73bc8470ed02a65ec4f4f81ec3ee41e2ab68d41c` on
`codex/miniwob-frozen-generalization-20261114`; experiment code was uncommitted
at evaluation time. `provenance-audit.json` therefore records SHA-256 for each
implementation file as executed. No media, weights, or large checkpoints are
added to Git.

## Interpretation and next gate

This result is useful evidence that a pretrained visual representation plus a
small option readout can solve this fixed four-choice synthetic screen task on
new simulator seeds, and that predictions depend on the correctly paired
screen rather than candidate position alone. It does not establish generic
image accuracy or a reusable world representation across tasks. It says nothing
about video, audio, text quality, or simultaneous-input fusion and does not
close R2 M-2. The high accuracy must not be generalized to the four-modality
95–98% product aspiration.

Keep this checkpoint frozen. The next roadmap decision should prioritize a
rights-cleared, genuinely both-input-required validation task and keep
Path A's other modality and fusion gates explicit. Do not tune this checkpoint
on the now-observed MiniWoB cohort, and do not promote it to a product model.
