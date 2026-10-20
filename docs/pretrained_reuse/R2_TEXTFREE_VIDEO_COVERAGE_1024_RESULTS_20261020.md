# Text-free CLEVRER coverage experiment results

**Status: both matched runs completed.** This is a narrow, video-only
development study. It tests whether doubling unique CLEVRER training coverage
helps a small readout over frozen VideoPrism frame features. It is not a test of
audio/video fusion, general multimodal quality, or product readiness.

## Execution and integrity

- Code branch: `codex/r2-clevrer-coverage-20261020`; source run commit
  `bbe04f774e57362a968e1ab9f0344e47602cd764`. The matching implementation is
  recorded at commit `6379bfe`; a report-accounting correction is at
  `f997ed6`.
- The `512` arm used 512 train scenes / 1,024 examples. The nested `1,024` arm
  used 1,024 train scenes / 2,048 examples. Both used the same 256 validation
  scenes / 512 sample IDs in the same order. Train/validation scene and media
  overlap were zero.
- Train manifest hashes: 512 `39ac25ea7012f40ed2c74b63e69ef4f265f035bec60b96c82bc219295dbd5f6d`;
  1,024 `61e4ca966dc431e3cce6832860ed54cc861945ab1e1876614367304eb0adadd3`.
  Validation hash for both: `2ff03047e7c763cadb975058b842e88102fa96d1327e4c2fa32dfef3c60b0cb5`.
- The scene/media split gate passed with zero overlap. The source's repeated
  normalized question/option templates were permitted as preregistered: the
  integrity receipt reports 130 cross-split template matches for the nested
  512 arm and 159 for the 1,024 arm. This is a limitation for claims about
  unseen question wording or answer-pattern transfer; those counts are not
  hidden or treated as new independent assets.
- Frozen feature cache: 1,280 unique videos × 8 positions × 768 features;
  SHA-256 `504f50303e1062344b1dcba3ad251974dc2c47d634572b1628582baf2396f5f5`.
  Extraction used JAX `TFRT_CPU_0`, took 1,640.5 seconds, and peaked at
  4,162,793,472 bytes working set. VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Training used the local NVIDIA GeForce RTX 3080 Laptop GPU, Python 3.11.9,
  PyTorch 2.6.0+cu124 / CUDA 12.4. Both arms used seed 17, AdamW, constant
  learning rate `1e-3`, weight decay `1e-4`, batch size 64, gradient clip 1.0,
  no early stopping, and exactly 2,048 updates per readout. Each readout saw
  131,072 example presentations. The 512 arm had 128 epochs and validated every
  32 updates; the 1,024 arm had 64 epochs and the same validation cadence.
- The 512 arm ran 2026-10-20 10:54:52–10:55:41 UTC (49.3 s); the 1,024 arm ran
  10:55:50–10:56:38 UTC (48.5 s). Per-readout training/evaluation time and
  peak CUDA allocation are in the run reports. The maximum readout allocation
  was 369,355,776 bytes. HotUGPU reserved 3,200 MiB and left 13,015 MiB free at
  child launch. Cloud cost: $0.
- Sealed audit and official test data were not opened. No previous Teacher,
  checkpoint, corpus, or artifact was modified. Model weights and media remain
  outside Git.

Two run attempts did not train: one failed before data loading due to Python
module resolution; another created only preflight files and stopped at update
zero when PowerShell 5 treated a PyTorch allocator warning on stderr as fatal.
Both output locations are preserved. The successful attempt used a corrected
launcher and a new output directory. Orbitune was allowed to yield and resume;
no process was stopped or signalled.

## Learning curves

The table reports macro metrics averaged over the two question types, with
equal weight for temporal and static descriptive questions. Values at every
listed step come from the same validation IDs. These are scheduled observations;
the selected checkpoint is separately chosen by the frozen minimum
question-type-by-taxonomy validation NLL rule.

| Updates | 512 MLP Acc / NLL | 512 BiLSTM Acc / NLL | 1,024 MLP Acc / NLL | 1,024 BiLSTM Acc / NLL |
|---:|---:|---:|---:|---:|
| 512 | 0.445 / 1.230 | 0.445 / 1.317 | 0.477 / 1.115 | 0.459 / 1.114 |
| 1,024 | 0.453 / 1.989 | 0.443 / 2.391 | 0.490 / 1.206 | 0.467 / 1.311 |
| 1,536 | 0.467 / 2.812 | 0.441 / 3.091 | 0.479 / 1.562 | 0.457 / 2.074 |
| 1,920 | 0.461 / 3.302 | 0.436 / 3.290 | 0.498 / 1.969 | 0.463 / 2.744 |
| 2,048 | 0.459 / 3.354 | 0.438 / 3.337 | 0.484 / 2.167 | 0.459 / 2.903 |

Training CE fell throughout. For the 512-scene MLP it fell from 0.7955 at step
512 to 0.0008 at step 2,048; validation NLL rose from 1.230 to 3.354, Brier
from 0.649 to 0.936, and ECE from 0.166 to 0.435. The 512-scene BiLSTM showed
the same pattern (training CE 0.6315 to 0.0006; validation NLL 1.317 to 3.337).
The 1,024-scene MLP also overfit (training CE 1.0277 to 0.0953; validation NLL
1.115 to 2.167). The 1,024-scene BiLSTM training CE fell from 1.0215 to 0.0160
while validation NLL rose from 1.114 to 2.903. These results show that 2,048
updates were more than the selected checkpoints needed; the final checkpoint
was retained separately and was not substituted for the validation-selected
best.

## Selected validation results

Each question type has 256 examples. Accuracy, NLL, Brier, and 15-bin ECE are
listed in that order.

| Train scenes | Readout | Best step | Macro Acc / NLL / Brier / ECE | Static descriptive Acc / NLL / Brier / ECE | Temporal descriptive Acc / NLL / Brier / ECE |
|---:|---|---:|---|---|---|
| 512 | MLP | 192 | 0.424 / 1.151 / 0.619 / 0.112 | 0.543 / 0.995 / 0.544 / 0.083 | 0.305 / 1.307 / 0.695 / 0.142 |
| 512 | BiLSTM | 288 | 0.471 / 1.136 / 0.608 / 0.115 | 0.578 / 0.963 / 0.524 / 0.100 | 0.363 / 1.308 / 0.692 / 0.130 |
| 1,024 | MLP | 672 | 0.502 / 1.093 / 0.589 / 0.083 | 0.594 / 0.891 / 0.494 / 0.048 | 0.410 / 1.295 / 0.683 / 0.118 |
| 1,024 | BiLSTM | 576 | 0.492 / 1.102 / 0.601 / 0.092 | 0.586 / 0.938 / 0.527 / 0.081 | 0.398 / 1.266 / 0.676 / 0.102 |

The MLP has 226,865 parameters and the BiLSTM 227,377. For both readouts, the
best and final checkpoints were saved separately. Best-checkpoint reload was
bitwise identical to the in-memory selected logits (maximum absolute
difference 0.0). Per-example probabilities and the complete validation
histories are in the external run directories.

## Paired scene comparison and interpretation

The 512 and 1,024 arms were compared on all 512 identical validation examples,
using 5,000 paired bootstrap resamples of the 256 underlying scenes, seed 17.
Intervals are percentile 95% intervals. The large-minus-small selected macro
Accuracy change was +7.79 points for the MLP (95% interval +3.91 to +11.52) and
+2.11 points for the BiLSTM (−1.76 to +6.05). For MLP temporal descriptive
Accuracy the change was +10.52 points (+4.69 to +16.41); static descriptive
was +5.05 points (−0.39 to +10.55). BiLSTM temporal change was +3.51 points
(−1.95 to +8.98), and static change was +0.71 points (−4.30 to +5.47).

This supports a useful unique-coverage effect for the MLP on this development
validation and suggests a temporal-descriptive gain. The BiLSTM interval is
compatible with little or no change, so the result does not establish an
architecture-independent gain. Both models remain far below the 95–98%
aspiration. The study covers only typed-program text-free CLEVRER descriptive
tasks and one seed; because the same validation chose checkpoints, the
bootstrap is diagnostic rather than blind confirmation. It does not establish
general video reasoning, AV complementarity, or any product-level quality.

Selected video ablations on the same validation show that the 1,024-scene
readouts use visual content for static descriptive questions, but do not show
reliable temporal-order use. For the MLP, static Accuracy was 0.594 aligned,
0.352 after scene-video shuffling, and 0.594 after reversing frame order;
temporal Accuracy was 0.410, 0.375, and 0.410. For the BiLSTM, static
Accuracy was 0.586, 0.344, and 0.609; temporal Accuracy was 0.398, 0.398, and
0.391. The paired scene-bootstrap aligned-minus-shuffled temporal Accuracy
intervals included zero for both readouts. Thus this run supports video
semantic contribution on static questions, while temporal sequence modeling
remains unproven. The repeated templates and validation checkpoint selection
further limit that diagnosis.

## Accounting correction and artifacts

The raw run reports used the legacy fixed `EPOCHS=256` constant for the
top-level repeat-presentation count, although their configured schedules and
actual optimizer updates were correct. The raw reports were left untouched.
The actual repeated presentations per readout are 130,048 for the 512 arm and
129,024 for the 1,024 arm, calculated from the effective epochs and unique
training rows. A separate `reporting-correction.json` in the external attempt
directory preserves both raw values and hashes. The runner was fixed for future
runs and tested; the correction changes no model or evaluation result.

External artifacts (`C:/CodexArtifacts/pretrained-reuse-r2/clevrer-coverage-1024-attempt-02/`):

- 512 and 1,024 run reports: SHA-256 `d9089eecbafa5b1fbc15d49babd303f7bceafc3320a9b271ac196cf132f34579` and `0c2ffbf8fdcae1a5dbfa2621c74a9561235bedc875730641a3871f6028e10972`.
- Matched paired comparison JSON: SHA-256 `71d0be4314dcc7a00851e140b2d5ba698c73b019c64e853534e166fe90359c87`.
- Reporting correction JSON: SHA-256 `5ffb13ab78c3b219d4ae016f3cd9860ae81c702651cd529ff08ed47fc05afb50`.
- Weight files, input preflights, effective configs, training histories and per-example predictions are preserved in each `run-512` / `run-1024` directory. No weight, cache, or media file was added to Git.

**Decision:** treat increased unique CLEVRER coverage as a promising video-only
training lever for the MLP, but do not promote this readout. The remaining
temporal accuracy is low, results are development-validation-selected, and the
project's multimodal and cross-task gates remain open. Next work should address
task/data validity and obtain a genuinely complementary audio-video dataset
before making a multimodal quality claim.
