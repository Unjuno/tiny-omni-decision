# HoloAssist AV + sensor readout and IMU interventions (2026-10-20)

## Scope

This is a narrow development-only diagnostic on HoloAssist action-quality
windows. It tests whether synchronized IMU features add useful signal to an
already trained video/audio temporal readout. It is not a general Omni quality
claim, a blind evaluation, or a product candidate. The frozen train and
development-validation split, AV features, AV checkpoints, Teacher artifacts,
and QAT artifacts were not changed. No official validation/test or sealed-audit
payload was loaded. All run weights and media remain outside Git.

## Data and implementation

The IMU source is the official synchronized HoloAssist archive. Only the same
1,042 training and 264 development-validation sessions used by the frozen AV
corpus were selected. The receipt covers 3,918 stream files; the source
selection hash is
`66f578b89ddb1757477b6c75e31b9d3fb7320d6832ae3d23dcd72ad67be17498`. IMU
features are 30 one-second bins per causal window, 21 values per bin, or 630
values per window. Normalization was fit on training features only and applied
unchanged to validation.

| Artifact | SHA-256 |
|---|---|
| Frozen AV train features | `1c080f3dd3a661ca8e3134754dcc9ef0f88a9ac8df53f9b2fdfed5561aafd2d3` |
| Frozen AV validation features | `2e760980460385cc615425f41e80d043fd66bc5e2d05c7ccf7d5742ee973b567` |
| IMU train features | `25e4d45a332df418e1934ee0a5e5d2a1ee72ef0ec7dfb1a3d884250ad1f628b5` |
| IMU validation features | `2a3b79e6ffb72815e5943ad2820d395a5ae874832d32f3417b2009db06ea082d` |
| IMU receipt | `5520afc81e28e1ee8f9675b284d23202b21b9cfc9b8daab8f61de32bb2118e41` |
| Feature-window receipt | `7e4862bae7701ccb8eaf4958e4efaa8079dad291dc094b2dec9f66dfa0b31475` |

The IMU join was checked against exact ordered validation IDs, targets,
loss-mask, session, window index, media identity, and task type. It produced
9,763 train windows from 1,042 sessions and 2,498 validation windows from 264
sessions, with zero train-validation session or media overlap. Of these,
9,351 train and 2,373 validation windows were labeled. No sealed data entered
the pipeline.

One initial IMU conversion attempt failed on repeated values in an auxiliary
integer timestamp column; it stopped before producing accepted features. The
parser was corrected to use the monotonic seconds timestamp as the ordering
axis while retaining the repeated auxiliary timestamp. The successful feature
build and its receipt were created afterward. One initial training attempt also
failed before model construction or optimizer updates because the config's
`hidden_size` was nested under `model`; that lookup was corrected and covered
with tests. It is retained as a failed attempt; the successful run used a new
output directory. The first pushed CI run passed lint but failed at test
collection because the intervention test imported the heavyweight PyTorch
runner in the dev-only CI environment. The session-derangement helper was
separated into a NumPy-only module; the measured intervention behavior is
unchanged and the CI suite is being rerun on that fix.

## Fixed-budget training

The run used source commit `c80551e9197efb22d3f32be2cc93f97a256e44e2`,
config `configs/pretrained_reuse/holoassist_av_sensor_temporal_readout_v1.json`
(SHA-256
`917afe75932a948093663e90e458868386216e733b0a1a799ab4e336aa37bbbd`), seed
17, hidden size 128, batch of eight sessions, AdamW at 0.001, weight decay
0.001, binary cross-entropy, and no early stopping. The three matched arms
were AV, sensor-only, and AV+sensor. Each used the same fixed train/validation
IDs and ran all 2,048 optimizer steps. Each arm encountered all 9,763 unique
training windows and 1,042 unique sessions; total consumption was 152,786
window occurrences, including 143,023 repeats. Best checkpoints were selected
by lowest validation NLL, ties kept the earliest step.

The run used the local RTX 3080 Laptop (16 GiB) under the cooperative HotGPU
reservation of 3,200 MiB. Run start/end were 2026-10-20 06:19:58.845 UTC and
06:25:46.998 UTC; all three arms together took about 5 minutes 48 seconds.
Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, PEFT 0.21.2, and PyAV
18.1.0 were recorded. Peak model-allocated CUDA memory was 91,496,960 bytes
(87.3 MiB), with 106,954,752 bytes (102 MiB) reserved. The HotGPU reservation
was honored; no process was stopped or modified. Cloud compute and paid
compute cost were zero.

Best validation checkpoint results (2,373 labeled windows):

| Arm | Best step | Accuracy | Balanced accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|
| AV | 960 | 0.7442 | 0.6667 | 0.5344 | 0.3530 | 0.0285 |
| Sensor only | 256 | 0.6987 | 0.5327 | 0.5941 | 0.4067 | 0.0366 |
| AV + sensor | 256 | 0.7370 | 0.6502 | 0.5498 | 0.3665 | 0.0188 |

The AVS model's selected gate means were video 0.4514, audio 0.3936, and sensor
0.1551 (per-window standard deviations 0.2750, 0.3090, and 0.2053). The gate
values indicate allocation by the learned model, not causal contribution by
themselves.

All arms completed the fixed budget, but their final checkpoints were worse
than their selected best in probabilistic quality. For AV, final NLL/Brier/ECE
were 0.6179/0.3853/0.1056; sensor-only 1.7949/0.6097/0.2794; AVS
1.6474/0.5151/0.2396. This is a clear late-training overfit/instability
signal, strongest in the sensor-only and AVS arms. Every selected checkpoint
passed reload parity at maximum probability delta `<= 1e-6`.
Each matched readout had 447,237 trainable parameters.

For the selected AVS-vs-AV paired session bootstrap (10,000 replicates, seed
1701), AVS minus AV had accuracy `-0.00722` (95% interval `[-0.01956,
0.00479]`), balanced accuracy `-0.01647` (`[-0.03265,-0.00120]`), NLL
`+0.01535` (`[+0.00252,+0.02740]`), and Brier `+0.01345`
(`[+0.00360,+0.02304]`). Sensor-only was also inferior to AV, especially in
sensitivity. Thus this task did not support an AVS fusion benefit at the
selected checkpoints.

## Fixed-checkpoint sensor interventions

The AVS step-256 checkpoint was loaded on CPU. Before intervention, its replay
was checked against the saved validation predictions in order and had maximum
probability delta `<= 1e-6`. No model weights, data labels, or checkpoint
selection were altered. Conditions were (1) baseline replay, (2) normalized
IMU set to zero, and (3) whole IMU sequences cyclically reassigned between
validation sessions matched on task type and exact session window count. The
derangement changed 213 of 264 sessions; the other 51 were singleton
task/length groups and stayed unchanged. This constraint preserves task and
sequence length but limits the intervention's coverage.

| Condition | Accuracy | Balanced accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Baseline replay | 0.7370 | 0.6502 | 0.5498 | 0.3665 | 0.0188 |
| IMU missing (zero) | 0.7181 | 0.6279 | 0.5742 | 0.3862 | 0.0319 |
| Same-task IMU reassigned | 0.7366 | 0.6452 | 0.5563 | 0.3718 | 0.0217 |

Paired session-cluster bootstrap (10,000 replicates, seed 1701, percentile
95% intervals) gives these candidate-minus-baseline differences:

| Intervention | Δ Accuracy | Δ NLL | Δ Brier | Δ balanced accuracy |
|---|---:|---:|---:|---:|
| IMU missing | -0.0189 `[-0.0298,-0.0081]` | +0.0245 `[+0.0133,+0.0360]` | +0.0197 `[+0.0105,+0.0289]` | -0.0221 `[-0.0341,-0.0107]` |
| Same-task IMU reassigned | -0.0004 `[-0.0091,+0.0082]` | +0.0065 `[-0.0004,+0.0135]` | +0.0052 `[-0.0005,+0.0111]` | -0.0050 `[-0.0147,+0.0050]` |

On these validation IDs, setting IMU to zero reliably degrades predictions,
showing the learned path uses the supplied sensor representation. But the
matched reassignment differences are small and their intervals include zero;
this does not establish strong session-specific IMU complementarity. The
zero-input effect may partly reflect an out-of-distribution intervention,
because zero after train normalization is not guaranteed to mimic naturally
missing IMU. Together with AVS trailing AV at its selected checkpoint, the
evidence is that this sensor path has measurable influence but useful added
information remains unproven for this task.

All intervals are validation-only and selection-biased because the same fixed
validation set selected the checkpoints. They are descriptive uncertainty,
not independent confirmation or a blind result.

## Reproduction and artifacts

The successful training command is retained in the external run metadata. The
intervention command was:

```powershell
python -m scripts.evaluate_holoassist_av_sensor_interventions `
  --run-metadata C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\avs-readout-attempt-02\run-metadata.json `
  --validation-features C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\packed-cache-v3\validation.npz `
  --imu-validation-features C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\imu-features-attempt-01\validation.npz `
  --imu-receipt C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\imu-features-attempt-01\receipt.json `
  --normalization C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\avs-readout-attempt-02\imu-normalization.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\avs-readout-attempt-02\sensor-interventions-cpu-attempt-01
```

Intervention output metadata SHA-256:
`6a7cf9a6e36a55da920b8748174ca8889a204c10ea4c430578a2ff21c5aa0dcd`.
Training metadata SHA-256:
`ad9e0c0dd83225f8b6e50057e09d855b526cea79ab97070aae5170d6ec664dfd`.
The final intervention evaluator SHA-256 is
`fe368fb2ce7aff87fd7206e653f82c39a0881b67d684d388c1cf118ee6c91621`; after
the run, only a behavior-preserving move of the NumPy-only derangement helper
was made so CI tests do not import PyTorch. Its tests are in
`tests/test_holoassist_av_sensor_interventions.py`.
The run, checkpoints, predictions, and all media/features remain in
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/`;
no large artifacts were added to Git.

The next useful study should improve sensor-data validity and task design
before expanding model capacity: confirm whether synchronized IMU is
semantically informative for the selected error labels, inspect coverage and
missingness, then test a predeclared sensor corruption or alignment control
without selecting on these same validation IDs. No further training, audit
evaluation, quantization, or product promotion was done in this continuation.
