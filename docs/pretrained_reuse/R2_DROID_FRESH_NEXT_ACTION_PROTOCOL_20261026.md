# DROID fresh-shard next-action protocol

## Scope

This is a development-only transfer check for the existing frozen-feature
Path A readout. It asks whether pretrained video features add information for
one-step Cartesian action direction beyond raw robot proprioception. It is not
a product-quality benchmark, a blind evaluation, or evidence for general
multimodal quality. No sealed audit was read.

## Pinned source and split

- Source: DROID 1.4.0, official train split, CC-BY-4.0.
- Fresh selection seed: 17; deterministic SHA-256 ranking after excluding the
  12 shards used by the earlier DROID work.
- Fresh shard IDs: `580, 1635, 1018, 717, 577, 1634, 487, 748, 1768`.
- The first ranked candidate, shard 654, was left out to retain disk reserve.
- GCS generation, size, and MD5 are pinned for all 9 shards and 3 metadata
  objects in the manifest. Download verification completed for 12 objects,
  428 episodes, 8,567,985,357 bytes.
- Selection manifest SHA-256:
  `cec05f0e1a39de3d7021e74ec390c745a0e31ce9a3d0763344864dcb2a2e34aa`.
- Download receipt SHA-256:
  `5d5a5ba5d59b8c76790db1c9399f6490e70098ad390efafe5ef7f94df63ee6f7`.
- Split unions source-file identity, recording-folder identity, and normalized
  instruction content. It excludes source/recording overlaps with the pinned
  DROID-100 debug audit. The deterministic seed-17 split targets 150 validation
  recording groups with at least 150 train groups.
- Prepared task: 275 train, 150 validation, 3 short episodes excluded; zero
  recording-group overlap. All seven action labels occur in train.
- Task-manifest SHA-256:
  `b7315c23bad4f9b195828c934cc3a87aeb34d36b1b1180a8e8152f783148e21d`.
- Effective config SHA-256:
  `a9564644fc0d9c8d011d6638a694d9d651da1d2dedbbec67c5b8f70d246f1c1b`.

## Task and model comparison

Each episode supplies 8 uniformly spaced states from its first 25% of steps.
The target is the dominant signed XYZ direction of commanded Cartesian
velocity at `last_input_index + 1`; `stop` applies when the largest absolute
translation component is at most `1e-4`. Options are `stop, x-, x+, y-, y+,
z-, z+`. The task uses exterior camera 1, exterior camera 2, and wrist video,
plus joint position, Cartesian position, and gripper position. The input does
not include future observations or the source success/failure path label.

Video uses frozen `google/videoprism-base-f16r288`, revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` (114.37M
parameters). Sensor features are raw 8-step sequences. Feature extraction is
CPU-only. Three L2 multinomial readouts are compared: video-only, sensor-only,
and concatenated video+sensor. Feature normalization is fit on train only;
regularization is selected by five-fold group-stratified macro NLL within train
using seed 17. Validation is evaluated once after the configuration is fixed.

## Artifacts and current stage

- Manifest: `manifests/pretrained_reuse/droid-fresh-next-action-seed17-20261026.json`
- Config: `configs/pretrained_reuse/droid_fresh_next_action_seed17_v1.json`
- Downloaded data and receipt:
  `C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-seed17-attempt-01`
- Prepared task:
  `C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-prep-attempt-02`
- Feature cache target:
  `C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-features-attempt-01`
- First readout fit (500-iteration convergence failure, no validation metrics):
  `C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-evaluation-attempt-01-fit-failure.json`
- Baseline readout result:
  `C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-evaluation-attempt-02`
- Stronger-L2 follow-up:
  `C:/CodexArtifacts/pretrained-reuse-r2/droid-fresh-next-action-evaluation-attempt-03`

CPU feature extraction completed for all 425 eligible examples in 1,452.45
seconds with `TFRT_CPU_0`. The first fit hit its 500-iteration cap during a
train fold, before computing validation metrics; the failure was preserved.
At 2,000 iterations the same L2 objective converged. All three arms selected
the upper end of the original L2 grid (100) on train-only CV, so a second
train-only CV configuration tested `[100, 1000, 10000]`. The development
validation was consequently scored more than once and is explicitly
nonblind. See the [results report](R2_DROID_FRESH_NEXT_ACTION_RESULTS_20261026.md).
The frozen VideoPrism representation and fitted readouts are diagnostic
artifacts, not a compact product model.

## Reproducibility commands

Run from the repository root. Dataset and experiment outputs remain outside
Git. The dedicated feature environment is the existing pinned VideoPrism
runtime; it is not upgraded by this experiment.

```powershell
python scripts/download_droid_fresh_cohort.py `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-seed17-attempt-01

python scripts/prepare_droid_fresh_next_action.py `
  --data-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-seed17-attempt-01 `
  --manifest manifests\pretrained_reuse\droid-fresh-next-action-seed17-20261026.json `
  --download-receipt C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-seed17-attempt-01\download-receipt.json `
  --debug-audit C:\CodexArtifacts\pretrained-reuse-r2\droid100-download-20261026-attempt-01\group-audit.json `
  --config configs\pretrained_reuse\droid_fresh_next_action_seed17_v1.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-prep-attempt-02
```

The extractor and evaluator commands are recorded with their result report
and are:

```powershell
C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe `
  scripts\extract_droid_fresh_video_sensor_features.py `
  --dataset-root C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-seed17-attempt-01 `
  --manifest manifests\pretrained_reuse\droid-fresh-next-action-seed17-20261026.json `
  --task-manifest C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-prep-attempt-02\task-manifest.json `
  --config configs\pretrained_reuse\droid_fresh_next_action_seed17_v1.json `
  --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz `
  --model-source C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-features-attempt-01

C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe `
  scripts\evaluate_droid_fresh_next_action.py `
  --task-manifest C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-prep-attempt-02\task-manifest.json `
  --feature-cache C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-features-attempt-01 `
  --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz `
  --config configs\pretrained_reuse\droid_fresh_next_action_seed17_v1.json `
  --readout-config configs\pretrained_reuse\droid_fresh_next_action_readout_fit_v1.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid-fresh-next-action-evaluation-attempt-02
```

The stronger-L2 attempt uses the same command with
`droid_fresh_next_action_readout_fit_v2.json` and output directory
`droid-fresh-next-action-evaluation-attempt-03`.
