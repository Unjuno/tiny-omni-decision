# DROID early-outcome frozen-feature readout — 2026-10-26

## Result and claim boundary

This is one development-only video/proprioception task. It does **not** close
R2 M-2, establish an audio/video task, or support a general Omni quality claim.
The target is the literal `success` / `failure` segment in the source path. It
is weak metadata supervision, not an independently annotated measure of task
quality. The source path is used only to create the target and stable record ID;
it is not a model input. Text and audio are absent from these experiments.

The fixed group-safe split has 337 train episodes (274 success, 63 failure)
and 200 development validation episodes (170 success, 30 failure). No sealed
audit was loaded. Models were not selected by validation; all three specified
readouts are reported. The evaluation uses the same frozen episode IDs and
options for each condition.

| Readout | Train accuracy | Validation accuracy | Balanced accuracy | NLL | Brier | ECE-10 | Failure recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train-prevalence baseline | — | 85.0% | 50.0% | 0.4275 | 0.1289 | 0.0369 | 0.0% |
| Video temporal | 100.0% | 76.5% | 54.6% | 1.1342 | 0.2000 | 0.1782 | 23.3% |
| Sensor sequence | 85.5% | 83.0% | 51.6% | 0.4925 | 0.1346 | 0.0546 | 6.7% |
| Video + sensor | 100.0% | 79.5% | 56.4% | 1.0908 | 0.1842 | 0.1597 | 23.3% |

All learned readouts miss the 85% accuracy of the train-prevalence baseline.
Video + sensor is 3.0 percentage points above video alone (class-stratified
paired 95% percentile interval: +0.5 to +6.0 points), but is 3.5 points below
the sensor-only readout; paired NLL and Brier differences versus video alone
include zero. Against the train-prevalence baseline, the fusion readout is
5.5 points lower in Accuracy (paired interval -10.0 to -1.5 points), with
higher NLL and Brier. Its 23.3% failure recall and large train/validation gap
show that this result is not a useful fusion success. The sensor-only head
mostly predicts success (6.7% failure recall), so its 83% Accuracy also masks
poor minority-class behavior.

This task is deliberately a weak screening proxy: the small validation set has
only 30 failures, and the labels are source-folder outcomes. Do not tune these
readouts further against this same validation set and call it an independent
result. The current evidence does not justify a trainable temporal gate or
promoting this task/model.

## Frozen features and readout

- Data: pinned DROID 1.4.0 train shards; 337 task rows for train and 200 for
  validation. The existing selected-shard and task-manifest checks remain the
  source of IDs, group split, labels, and eight early observation indices.
- Video: released VideoPrism-B, revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`; code commit
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  The full extraction used CPU `TFRT_CPU_0`, 8 early frames × 3 views per
  episode, the upstream resize-short-side/center-crop preprocessing, and
  spatial token mean pooling. It produced `[537, 3, 8, 768]` float32 features.
- Sensors: eight aligned time steps × 14 values (7 joint position, 6 Cartesian
  position, 1 gripper position), finite float32; no future observations.
- Video readout: average the three views, then preserve separate early and late
  four-frame means (1,536 values). Sensor readout flattens its eight-step input
  (112 values). Fusion concatenates those vectors (1,648 values); it has no
  learned gate. Each condition uses a deterministic L2 logistic head; feature
  scaling is fit on train only. No architecture or regularization search was
  performed.
- The video/fusion readouts each have 1,537 / 1,649 trainable scalars including
  intercept; the sensor-only readout has 113. With the existing
  114,365,184-parameter VideoPrism checkpoint, the video/fusion totals are
  114,366,721 / 114,366,833 parameters. The checkpoint file alone is
  457,475,296 bytes; the fused readout file is 41,313 bytes. The combined
  minimum weight payload is 457,516,609 bytes, above the provisional 200 MB
  package target. This is storage arithmetic, not a complete packaged-runtime
  measurement.

## Execution and artifacts

The feature extraction ran 2,068.2 seconds on the local CPU. No GPU, cloud
compute, or paid resources were used. Peak RAM for the full extraction was not
recorded; a separate one-episode smoke previously measured 4,259,770,368 bytes
peak working set, and that figure must not be presented as the full-run peak.
The three readouts took about 0.65 seconds of optimizer time in total; full
readout evaluation, paired bootstrap, serialization, and reload checks took
2.85 seconds. All three readouts converged. Save/reload was checked against the
same validation probabilities with exact equality.

Pinned artifacts are stored outside Git:

- Frozen features, extraction progress, and summary:
  `C:\CodexArtifacts\pretrained-reuse-r2\droid-videoprism-early-outcome-features-attempt-03\`
- Readouts, per-example validation predictions, and evaluation report:
  `C:\CodexArtifacts\pretrained-reuse-r2\droid-early-outcome-readouts-repo-attempt-02\`
- Feature cache hashes: video
  `d97a7f28bcdc119cccf4ef0d2019ae69852734b43a10d9102d320c0ff3043878`, sensors
  `6ef3031c5521d1e40d8f19dd527e0709abeab0af44748f6a90416c15316d1016`, labels
  `ac80dc7a7ab52ef31c9a048ed29bfa15525cd14b161934a44c113738d68bcc46`.
- Evaluation JSON SHA-256:
  `a37512bd3e122d69db761e8acdc25c388609b0ca26eba95402caddac257b3a91`.
- Readout SHA-256: video
  `77deb517dca9d5e9073f0524c947b1d5190b8868a2f557590c96b402d71019e6`, sensor
  `6f9688b5f9f569da64d9bdcab0838d0c9a0aaf55c8b2f28e2de973ab514b74a0`, and
  video+sensor
  `0c506139e1301dbaa15889220a3d89dc7924d06d1307a418919d8e1a9a6c50eb`.
- Per-example prediction SHA-256: video
  `6fb395023e891f9aa1f331c3a7eeea674be4ae73d7b16de575dfce6a0f152ea5`, sensor
  `ad1ae0b05d2ff57a10e646f6116174a68887447b95c7c08562bd87aaf9b4463d`, and
  video+sensor
  `639f54cddaac7a733735abf5915f8f9740f40989afabbd132d7046286bccf58b`.
- The initial one-off evaluation attempt wrote hard labels into its per-example
  probability field. That attempt is preserved and **not used** for the metrics
  above. Correct probabilities were regenerated from the saved readouts; the
  repository evaluator now writes the actual probabilities and refuses to
  overwrite output directories.

The extraction driver for this run is preserved outside Git at the feature
attempt directory; its SHA-256 is
`efe6e4c9e375a57dd6af698797c448c2fff20ead30d2d16d9a53453fccf65490`. The
source/task pipeline and frozen feature run started from repository commit
`4b5742d14aafe7a2c0b60351a940483d53a75acc`; the readout evaluator, metrics,
tests, and this report are recorded in commit
`a40b526ec3fa37db87e78f1ad4cbd349ae110c37` on
`codex/r2-droid100-source-task-screen-20261026`. The
reusable readout/evaluation implementation is
`scripts/evaluate_droid_early_outcome_readouts.py` and
`src/tiny_omni_decision/droid_early_outcome_readout.py`; both remain subject to
the current branch's code review and CI. Their SHA-256 hashes are
`1181c24ebedc04071936ee2636a2a728e1549795141a16832be5ed615db83740` and
`80cd9adde78d21a0bc67d38cf8e21c9c5d991d7d2fd95205dc32b39cd79e1c34`.
Focused DROID and readout tests pass
with SciPy present (20 passed). The ordinary project interpreter lacks SciPy,
so its default test invocation skips the optimizer smoke; the actual fit ran
in the pinned VideoPrism environment with SciPy available.

The repository evaluator invocation was:

```powershell
$env:PYTHONPATH = 'C:\CodexWorktrees\r2-droid100-source-task-screen-20261026\src'
& 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  scripts/evaluate_droid_early_outcome_readouts.py `
  --cache-dir 'C:\CodexArtifacts\pretrained-reuse-r2\droid-videoprism-early-outcome-features-attempt-03' `
  --task-dir 'C:\CodexArtifacts\pretrained-reuse-r2\droid-early-outcome-v0-attempt-02' `
  --checkpoint 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\droid-early-outcome-readouts-repo-attempt-02'
```

## Disposition

The feature extraction and fixed readout comparison are complete. The result
does not justify model promotion or a claim of useful video/sensor fusion.
R2 M-2, independent task labels, failure-class coverage, synchronized AV,
four-modality quality, production package/runtime validation, and the broader
R0/R1/R3–R8 gates remain open. Existing Teacher artifacts, prior model runs,
sealed audits, and legacy worktrees were not changed.
