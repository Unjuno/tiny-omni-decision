# R2 MiniWoB frozen-checkpoint prospective screen-only evaluation — 2026-11-14

## Scope and disposition

This records the already completed, one-time evaluation in the preserved
`miniwob-frozen-generalization-20261114-attempt-02` output. The run used a
fixed DINOv2-small screen-only click-color scorer and performed no training,
calibration, or checkpoint selection. It did not load a sealed audit. This is
prospective development/generalization evidence for one generated UI task; it
is not evidence for natural-image understanding, video reasoning, multimodal
fusion, or product quality.

The run began from a previously selected checkpoint, so its result must be
interpreted with the known selection history: the scorer was selected using an
earlier 512-row development validation set. This fresh evaluation does not
erase that history. Preserve these 1,024 episodes as exposed development data;
do not tune against them and then call them held out again.

## Frozen source, cohort, and checkpoint

- MiniWoB++ source: `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`, MIT.
- Experiment code commit recorded in run metadata:
  `73bc8470ed02a65ec4f4f81ec3ee41e2ab68d41c`.
- Experiment ID: `r2-miniwob-dino-frozen-generalization-v1-seed17-attempt02`.
- Seed window: `[1008192, 1016384)`; 1,805 seeds were attempted and 1,024
  valid unique episode groups were accepted. Each accepted sample had a unique
  seed and screenshot pixel identity.
- The pre-generation receipt compared this seed window with 12,329 historically
  used seeds and 2,145 earlier screenshot identities. It recorded zero seed
  overlap and zero screenshot-pixel overlap. Its SHA-256 is
  `7f3668a66d96fdfaca8edf816086cd2283891e6e276ed09792184fb97bcdb568`.
- Evaluation JSONL SHA-256:
  `5b5203ef8a8afbbafe1e41ed66196ffa362e548ef7f4a6b0309408651496d87e`.
- Evaluation order SHA-256:
  `9311dd03011f95029c8a4145df82156bd19a9a680aefc9bb5cc0c5ec1da7378d`.
- Frozen encoder: DINOv2-small revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, 22,056,576 parameters, weight
  SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- Fixed option readout: 76,293 parameters, SHA-256
  `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`.
  Resident image-path parameters total 22,132,869. No weights changed.
- Effective config SHA-256:
  `f85740038323344acb311a62bc42982f08f751a9e4285a727f237878f93f990e`.
- Corpus manifest SHA-256:
  `1a7b35e681677739bac456ff8a5fbba3cfbb4ba09da1e3d6ed6f501f4f2df84d`.

The first collection attempt had a finalization/import failure. Attempt 02
reused and hash-verified the already preserved screenshots in a new output
directory; it did not overwrite them. The recovery receipt reports 1,024
preserved screenshots and exact pixel identity. The scorer was loaded only for
the subsequent fixed inference pass.

## Fixed results

Metrics are over the same 1,024 accepted episodes. Four-choice chance accuracy
is 25%.

| Scenario | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Matched screen | 0.9697 | 0.1223 | 0.0594 | 0.0511 | 0.9212 |
| Train-derived position prior | 0.2402 | 1.3881 | 0.7509 | 0.0273 | 0.2676 |
| Black screen | 0.2588 | 2.3441 | 0.9319 | 0.3360 | 0.5948 |
| Cross-episode Sattolo screen shuffle | 0.2490 | 5.2159 | 1.3725 | 0.6722 | 0.9212 |
| Candidate-order remap | 0.9697 | 0.1223 | 0.0594 | 0.0511 | 0.9212 |

The candidate-order remap is a software invariance control, not a separate
quality sample. After mapping by stable option identity, maximum probability
difference was `1.1921e-7`; the predeclared tolerance was `2e-6`.

The episode-group bootstrap for matched screens used 10,000 resamples and seed
17. Its percentile 95% intervals were Accuracy `[0.9590, 0.9795]`, NLL
`[0.1051, 0.1412]`, Brier `[0.0482, 0.0717]`, and ECE-15 `[0.0433, 0.0616]`.
Matched-minus-control paired intervals excluded zero for the position-prior,
black-screen, and shuffled-screen controls. They were effectively zero for
candidate-order remapping, as expected.

The earlier 512-row development validation scored 0.9609 Accuracy, 0.1231 NLL,
0.0617 Brier and 0.0506 ECE-15. The new sample's 0.9697 accuracy is a
descriptive `+0.0088` difference only; do not interpret it as a clean estimate
of a performance increase because the earlier set participated in checkpoint
selection and the samples are separately generated.

## Execution and reproducibility

- Recorded run interval: `2026-11-14T03:13:57.984093+00:00` to
  `2026-11-14T03:22:49.398175+00:00`.
- Device: CPU; CUDA was not initialized. Cloud cost: `$0`.
- No model training or calibration; no checkpoint selection; sealed audit not
  loaded.
- Prediction JSONL SHA-256:
  `ea8cd018f876a1bedd77531556658097b2ae3ff0b17c8b3fef7762d0b81988a3`.
- Results JSON SHA-256:
  `c4eef34de653ebe382757523834ee9109513875f0e243c24741e9fa7d709304d`.
- Run-metadata SHA-256:
  `ce9b0c2125947f633ce0677960c5a31081255818f401735094377c4b8b74b294`.
- Output directory:
  `D:\CodexArtifacts\pretrained-reuse-r2\miniwob-frozen-generalization-20261114-attempt-02`.

Hashes were recomputed from the preserved files before writing this report and
match the recorded metadata. No earlier MiniWoB corpus, checkpoint, or result
was changed.

## Interpretation and next gate

The screen association is necessary for this synthetic click-color task: the
matched-screen result is far above the position, black-screen, and shuffled
screen controls. This does not demonstrate that the encoder understands
general images, that it reasons over video, or that different modalities
complement one another. The baseline also has known checkpoint-selection
history and remains a narrow one-task result.

Keep the checkpoint as a screen-only candidate and move the main R2 effort to
independently rights-cleared multimodal tasks, source-grouped splits, and
both-input-required controls. Do not create another MiniWoB test on these same
seeds or reuse this cohort for tuning.
