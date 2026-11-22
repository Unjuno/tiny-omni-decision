# R2-M HapTile Temporal Decision Pilot Results — 2026-11-22

## Status and scope

The frozen-feature CPU pilot completed all six predeclared input arms for 80
epochs each. It is an exploratory, task-held-out Decision experiment using
HapTile text, robot state, front/wrist video, and tactile video. HapTile has no
audio; this run does not evaluate the required text/image/audio/video model and
does not meet the project completion criteria. It is not a product candidate.

The frozen dataset contains 1,113 train episodes before the label-blind video
window gate. One episode lacked enough distinct causal frames and was excluded
by the predeclared gate, leaving 1,112 train examples. Validation has 169
examples across five held-out task groups; calibration has 169 examples across
four distinct task groups. The sealed audit was not loaded. No quantization was
performed.

## Implementation correction

Full feature assembly first failed closed because its provenance check required
all pre-gate state/text rows, while the already-frozen train state/text
extractors had correctly emitted only the 1,112 gate-eligible rows. The
assembler now accepts either the complete frozen IDs or exactly the eligible
IDs, checks each row against the action metadata, and rejects any other subset.
It also accepts the train-only state's explicit
`validation_or_sealed_audit_opened: false` proof while requiring its
train-only extraction status. Regression tests cover both forms and reject
ambiguous/development provenance. The original frozen data and feature arrays
were not modified.

## Reproduction record

- Repository: `Unjuno/tiny-omni-decision`
- Source commit: `d5b83c0d6dcf0281f560a9432ca1b674de1e72fb`
- Config: `configs/pretrained_reuse/haptile_temporal_decision.yaml`
- Config SHA-256: `c0003bf98d8182e945fc45b25378f0bec5ba3b0889b7bdae89eafce08c284f80`
- Executed trainer SHA-256: `290c7a175e5ed2d8fe97399b46d83f39c0fc82e9874efc1ba0cc4e9dee0d4e04`
- The feature assembler needed an uncommitted fix before the joined arrays
  could be built; its exact executed SHA-256 was
  `fc714b47312a3c99fd8d5ca65b8f6021b049749df1a76b67f7a9533307d0adc1`.
  That tested fix is included in commit `fe427691`, which records this result;
  it was not part of the `source_commit` below. The training script itself is
  byte-identical to the recorded source revision.
- Command: `py -3.11 scripts/pretrained_reuse/train_haptile_decision.py --artifact-root C:\\CodexLocalArtifacts\\pretrained-reuse-r2\\haptile-task-v1 --output C:\\CodexLocalArtifacts\\pretrained-reuse-r2\\haptile-task-v1\\fit-seed17-v1-attempt-01 --config configs/pretrained_reuse/haptile_temporal_decision.yaml`
- Start/end: 2026-11-22 03:15:47.813–03:31:19.322 UTC (15m 31.5s)
- Device: CPU, Intel Family 6 Model 154; 4 torch threads; Python 3.11.9,
  PyTorch 2.6.0+cu124, NumPy 2.4.6; CUDA unused; paid compute cost 0.
- Frozen VideoPrism-B revision: `be719a406d563b66f0ac969e7c94bab8e997c81a`,
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`,
  114,365,184 parameters.
- Frozen MiniLM-L3 revision: `4ca70771034acceecb2e72475f72050fcdde4ddc`,
  weights SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`,
  17,389,824 parameters.
- Train/validation/calibration joined feature SHA-256:
  `4d0b8bfc4d9d279ec04e76811f6a9fc6a8410ceb3d498a33cca9cc0112ba42c`,
  `ed874a01274e23e6ac8da4645f5bc2a46aca1422df7ca3145c8ddb33c797fa44`,
  `a6659c6e43d7c2d09242b493cb7eede7977d09703b776719e37c58a7078b8acb`.
- Train/validation/calibration counts and task groups: 1,112/24; 169/5;
  169/4. Each row uses a unique underlying media asset in its split.
- Every arm completed 80 epochs and 2,800 optimizer updates. Six arms total:
  16,800 optimizer updates. All best checkpoints passed save/reload parity.
- External run directory:
  `C:\\CodexLocalArtifacts\\pretrained-reuse-r2\\haptile-task-v1\\fit-seed17-v1-attempt-01`
  (weights, histories, predictions and run metadata stay outside Git).

## Best validation checkpoints

Checkpoint selection used the minimum validation macro task-group NLL within
each arm. The candidate arm was then selected by the same validation-only rule.
Metrics below are from each arm's best checkpoint; Accuracy is plain example
Accuracy and `macro` metrics weight the five validation task groups equally.

| Arm | Best epoch | Best validation macro NLL | Macro Accuracy | Macro Brier | Macro ECE | Last epoch macro NLL |
|---|---:|---:|---:|---:|---:|---:|
| text only | 1 | 1.712 | 40.6% | 0.774 | 0.194 | 1.794 |
| state only | 12 | 1.430 | 47.4% | 0.627 | 0.228 | 4.287 |
| vision + text | 5 | 1.492 | 42.4% | 0.681 | 0.242 | 5.337 |
| tactile video + text | 11 | 1.731 | 42.2% | 0.772 | 0.275 | 3.947 |
| vision + tactile video + text | 11 | 1.441 | 47.3% | 0.638 | 0.223 | 6.678 |
| full (vision + tactile video + state + text) | 19 | **1.120** | **63.1%** | **0.508** | **0.197** | 4.199 |

For the selected full arm, example-level validation metrics were 61.5% Accuracy,
1.164 NLL, 0.525 Brier, and 0.103 ECE. Macro task-group metrics were 63.1%
Accuracy, 1.120 NLL, 0.508 Brier, and 0.197 ECE. The selected epoch was 19 of
80; the last epoch's macro task-group NLL was 4.199. The growing train/validation
gap across the arms is direct overfitting evidence, not a reason to relabel the
fixed validation set as a test set.

The full arm's validation-selected single-stream comparison was state-only
(47.4% macro Accuracy, 1.430 macro NLL). A 2,000-draw, seed-17 paired task-group
bootstrap reported a full-minus-state macro Accuracy difference of +15.7
points (95% percentile interval +7.3 to +24.1 points) and macro NLL difference
of -0.311 (interval -0.414 to -0.208). The bootstrap uses only five task groups,
and arm selection used this same validation set, so these intervals are
selection-affected development diagnostics, not independent significance or
generalization evidence.

## Input-dependence controls and calibration

On the selected full model, validation macro NLL was 1.120. Replacing video with
zeros raised it to 1.650; replacing the robot-state/tactile sequence raised it
to 2.662. Shuffling video within task groups raised it to 2.170; shuffling
sensor/state streams raised it to 2.896. These controls show sensitivity to
the supplied streams in this pilot. They do not establish robust cross-modal
reasoning or out-of-distribution utility.

After validation-based checkpoint selection, one scalar temperature was fitted
on the separate 169-example calibration split using four different task groups.
The fitted value was 1.8067. Its in-sample calibration macro NLL changed from
1.540 to 1.365; on the validation predictions, temperature scaling changed
macro NLL from 1.120 to 1.115 and did not change Accuracy (63.1%). Calibration
metrics are in-sample for the calibration split and must not be presented as
unbiased test calibration.

## Capacity and deployment implications

The selected full Decision head has 519,683 trainable parameters (about 2.88 MB
for its saved checkpoint). Adding the frozen VideoPrism-B and MiniLM-L3
parameter counts gives about 132,274,691 parameters for the tested representation
stack plus head. This is within the provisional 100–200M parameter range, but
the VideoPrism checkpoint alone is 457,475,296 bytes, before MiniLM, runtime
code, or other files. The tested stack therefore does **not** meet the
provisional 200 MB package target. No quantized deployment package, complete
resident runtime, inference latency, or RAM/VRAM budget was measured.

## Conclusion and next evidence

This run establishes that frozen text, video, and sensor representations can be
joined and that a small Decision readout can train, save, reload, produce
probabilities, and respond to stream-ablation controls. On this fixed
task-held-out validation split, the full model outperformed the best unimodal
arm by the reported metrics. It also overfit strongly after its selected epoch.
With only five validation task groups, one seed, and no independent test result,
the gain is preliminary. It must not be promoted as a general-purpose Omni
Decision model.

Next, preserve this run and implementation, add a genuinely independent set of
task groups/source families with verified labels and paired-input necessity,
and repeat a predeclared grouped evaluation. Image, audio, cross-domain
generalization, quantization, the <=200 MB runtime package, and target-device
latency remain untested. Do not open the sealed audit for tuning.
