# R2 one-time frozen image readout on official CLEVR-4 validation

Date: 2026-10-29 JST  
Source branch commit: `86d45fb0e8adcbf2104aff99171b1cf898a1e31e`  
Evaluation output (external, not committed): `C:\CodexArtifacts\pretrained-reuse-r2\textfree-image-official-val-onetime-20261029-03`

## Protocol and provenance

This is a one-time, CPU-only evaluation of a frozen text-free image taxonomy readout. It used 2,107 official CLEVR-4 validation images, with four 10-way questions per image (8,428 decisions). Cached VideoPrism-B features were computed earlier; this evaluation performed no encoder forward, training, calibration, prompt search, or checkpoint selection. Sealed audit was not loaded.

The readout was selected at epoch 502 by minimum macro NLL on its fixed internal validation set. Its source fit report records `official_validation_loaded=false`, `sealed_audit_loaded=false`, 7,582 unique train images, and 842 internal validation images. Before evaluating, the script verified the fit report against its `run-report.sha256` sidecar, the selected checkpoint and feature-cache hashes, exact validation row order/IDs/media hashes, and zero ID/media SHA overlap between the readout's training partition and the evaluated split. All four JSON Decision candidate-order checks passed.

The official validation set has been accessed by other project experiments. Therefore this is **not blind project-level evidence** and it must not be reused for tuning. The synthetic CLEVR-4 taxonomy task is not evidence of general natural-image reasoning or joint multimodal decision quality.

Command:

```powershell
python scripts/evaluate_textfree_image_official_validation.py `
  --fit-dir 'C:\CodexArtifacts\pretrained-reuse-r2\textfree-image-decision-512-01' `
  --feature-cache 'C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-videoprism-features-v0\features.npz' `
  --corpus-dir 'C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-coverage-seed17' `
  --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\textfree-image-official-val-onetime-20261029-03'
```

## Results

Each taxonomy has n=2,107. Macro rows are unweighted means across the four taxonomies.

| Taxonomy | Accuracy | NLL | Brier | ECE (15 bins) | Majority baseline |
|---|---:|---:|---:|---:|---:|
| Color | 99.86% | 0.0086 | 0.0034 | 0.0056 | 10.92% |
| Texture | 95.59% | 0.1577 | 0.0708 | 0.0415 | 10.63% |
| Count | 59.33% | 0.9277 | 0.4998 | 0.0317 | 10.58% |
| Shape | 99.95% | 0.0020 | 0.0008 | 0.0015 | 12.67% |
| **Macro** | **88.68%** | **0.2740** | **0.1437** | **0.0201** | — |

The internal development score was 88.30% macro accuracy (count 58.19%); the official validation score is 88.68% (count 59.33%). The small difference is descriptive only. Count is the clear weakness: exact count accuracy is 59.33%; 94.02% of predictions are within one object, with mean absolute error 0.472. A posthoc expected-count/rounding calculation gave 61.60% exact and MAE 0.416; this alternative rule was **not adopted or selected** and is recorded only as error characterization. Do not tune on this split.

## Model and compute

- Frozen encoder: `google/videoprism-base-f16r288`, revision `be719a406d563b66f0ac969e7c94bab8e997c81a`, 114,365,184 parameters.
- Readout: 30,760 trainable parameters; frozen encoder plus readout: 114,395,944 parameters.
- The readout itself took 0.0114 seconds on CPU using cached features. No CUDA was used. This is not end-to-end image inference latency.

## Hashes

- Effective fit config: `c4118b5026a52bd570fa9f1e7d94c8d47e313a415ebeef3412be8a0b2bc7e0fc`
- Fit report: `f2b357b00bf2ce11cc3c1b6f75b64aa8ea1a9c132908601c1fafc2e98fe80b39`
- Selected readout: `e62711ca1b8fd68df5037cd135266ed7f556e0ad006359305f10e6665789a47d`
- Feature cache: `f36b9cf17efed6a5137eb48ffcaf40ff960015fc07bc194fb532bf5562d88fab`
- Validation labels: `3db398bc2f5658f34891a5687ea44193da9529854695fab7b343f36a1bbb8adb`
- Validation media manifest: `7b3c1b5ef0698e5eeb967b85d2a6beac6b8cc5792f45178d56fc986511ba2d08`
- Evaluation report: `792908a3d189b9f58683e682ec6cfad8f3afff3ebe12b99365afe06801a5356a`
- Per-example predictions: `ef189b3821c6a16e9cf7ee3e7a53ee2fbe247b1aed3ed56b1580dd8748f7f88a`
- Score tensor: `df65c9737175548af06f3e57e7cc405b85f93710d44a30f956c86a3787c8a007`

The evaluation JSON also records source hashes, package versions, IDs/order hashes, metrics, and limitations. The source archive-wide SHA-512 was not verified; per-image media hashes and feature-cache hash were checked.
