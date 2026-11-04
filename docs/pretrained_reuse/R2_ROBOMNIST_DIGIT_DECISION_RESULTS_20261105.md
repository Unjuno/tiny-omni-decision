# RoboMNIST digit Decision — 2026-11-05

## Result

A ten-option task-native Decision was trained on the frozen audio/video encoder
features: identify which digit the robot drew. Folds held out complete
robot-arm × speed × digit conditions. Chance Accuracy is10%. At the best fixed
report point (step25), joint audio+video reached20.64% Accuracy, Audio-only
18.72%, and Video-only10.90%. Yet their NLLs are poor (joint12.929, audio5.351,
video10.987), with severe overconfidence. At400 steps all three models reached
100% fit accuracy in each fold while OOF condition accuracy remained11.54–19.74%
and NLL worsened. This shows strong condition-level overfitting in this
readout/source setup; it does not prove the encoders or architecture cannot
improve with other valid data/tasks.

## Learning curve

|Modality / step|Accuracy|NLL|Brier|ECE-15|Mean confidence|
|---|---:|---:|---:|---:|---:|
|audio_only / 1|0.108|3.101|1.064|0.325|0.433|
|audio_only / 25|0.187|5.351|1.293|0.576|0.764|
|audio_only / 50|0.187|5.929|1.328|0.608|0.796|
|audio_only / 100|0.185|6.194|1.342|0.620|0.805|
|audio_only / 200|0.190|6.476|1.355|0.626|0.815|
|audio_only / 400|0.195|6.862|1.370|0.636|0.830|
|video_only / 1|0.029|5.867|1.400|0.579|0.608|
|video_only / 25|0.109|10.987|1.575|0.744|0.853|
|video_only / 50|0.110|11.489|1.592|0.757|0.867|
|video_only / 100|0.110|11.526|1.590|0.757|0.868|
|video_only / 200|0.112|11.545|1.590|0.757|0.869|
|video_only / 400|0.115|11.592|1.592|0.756|0.871|
|joint / 1|0.064|6.486|1.424|0.637|0.701|
|joint / 25|0.206|12.929|1.464|0.701|0.907|
|joint / 50|0.203|13.536|1.484|0.714|0.917|
|joint / 100|0.199|13.597|1.488|0.719|0.918|
|joint / 200|0.196|13.598|1.491|0.721|0.917|
|joint / 400|0.197|13.604|1.495|0.718|0.916|

Per-class OOF Accuracy at step25 (digit0–9):
|Modality|0|1|2|3|4|5|6|7|8|9|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|audio_only|0.41|0.50|0.03|0.08|0.18|0.00|0.03|0.13|0.50|0.00|
|video_only|0.24|0.12|0.05|0.04|0.00|0.04|0.08|0.45|0.01|0.04|
|joint|0.46|0.64|0.18|0.08|0.00|0.00|0.03|0.38|0.28|0.00|

The uneven class recalls and severe NLL/ECE reject this as a useful decision
model. The joint classifier is only a small gain over audio at step25, and its
calibration is much worse; this is not evidence that the task requires both
inputs. No model is selected or promoted.

## Integrity, model and compute

- Frozen protocol: [task and split design](R2_ROBOMNIST_DIGIT_DECISION_PROTOCOL_20261105.md).
- Source commit `8c7fc9a575879617f905aa962e461eba3372838a`; config SHA-256 `4beae402952d13edc1711112823a1941b6fd1b9b60545005bf6960e415d80d00`;
  prediction SHA-256 `11108dc5cbb85fb029cef56a08d61c5a01e9fa48584bac5a90135506e02be46d`; frozen manifest SHA-256
  `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- 780 original TRAIN repetition groups across60 conditions, stratified into
  five folds by full robot×speed×digit condition; 12 held conditions per fold.
  All mean/std transforms fit only within fit-fold groups. Original validation
  feature arrays and sealed audit were not opened. These source groups have
  prior project development exposure, so this is not blind evidence.
- Options are fixed IDs0–9; option-order mapping is covered by tests. Labels use
  Ac solely as target; split metadata and filenames are not model inputs.
- Readouts have7,690 Audio /15,370 Video /23,050 joint trainable parameters.
  Combined with frozen encoders, AV resident parameter arithmetic is
  116,349,473 for joint. This is not a four-modality product package count.
- All90 scheduled checkpoints reload with exact probability parity. Runtime
  11.82s; peak process working set
  604,454,912 bytes. CPU only, no cloud/GPU cost.
- Exact command: `python scripts/fit_robomnist_digit_decision.py --manifest C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-match-split-20261103-attempt-03/manifest.json --cache C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-match-feature-cache-20261103-attempt-01 --output C:\CodexArtifacts\pretrained-reuse-r2\robomnist-digit-decision-20261105-attempt-01`.
- New artifacts remain external at `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-digit-decision-20261105-attempt-01` with effective config, fold
  normalizers, per-fold history, class probabilities and checkpoints.
- Machine-readable metrics and class counts:
  [JSON](R2_ROBOMNIST_DIGIT_DECISION_RESULTS_20261105.json), [CSV](R2_ROBOMNIST_DIGIT_DECISION_RESULTS_20261105.csv).

This is a narrow synthetic action-classification development task; it does not
close R0/R2, establish world-model reuse quality, or meet the requested
multimodal product target. The negative result is retained without further
selection on these same OOF conditions.
