# iNaturalist new-observer transfer result (2026-10-29)

## Result scope

One-time development-transfer evaluation of the already-selected VideoPrism-B frozen image readout on the 176-record observer-disjoint CC0 validation cohort. This is a fixed eight-species image-classification probe; it is not an open-vocabulary Decision benchmark, sealed audit, four-modality result, or product-readiness claim. No training, checkpoint selection, calibration, or hyperparameter search occurred. The separate 700-train / 200-validation expansion-training gate remains failed (430 / 176 eligible rows).

## Frozen inputs and integrity

- Filtered validation JSONL SHA-256: `d5028e54f7d0000c0936f7319f6f8a598f6c458be02f10b2555ea7baf74729b4` (176 rows, fixed order).
- Downloaded media manifest SHA-256: `1b4bbdd20c093a5a55410e30853e95a12e9b04719620730c3d0caa4bc705b0ab`.
- Feature cache SHA-256: `36f3d50843bc81ea10e8c1190453150044a472e8f6eec85cd4e5032d6ec5276d`.
- Selected readout SHA-256: `323d0954a83c037ba5c6b37da3997addba48a66787a072acd3bae5fdbe7d2b88`.
- VideoPrism revision `be719a406d563b66f0ac969e7c94bab8e997c81a`; upstream source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- Runtime: Python 3.11.16, JAX 0.6.2, TensorFlow 2.18.1, NumPy 1.26.4, `TFRT_CPU_0`; CUDA hidden. Encoder parameters: 114,365,184; feature width 768.
- All 176 media files are present, explicitly CC0 at observation and photo level, with finite features and matching sample ID order. Against the 1,642 old train and 683 old validation images: exact raw/media/normalized-RGB overlap counts are all 0; pHash distance <=4 count is 0. No sealed audit was read.

## Runtime and storage

The first three sorted assets averaged 215,703 downloaded bytes and 345,205 normalized bytes per image, taking 4.843 sec total (1.61 sec/item). Full acquisition used 27,128,015 downloaded bytes and 60,046,213 stored bytes, taking 242.0 sec. Frozen CPU feature extraction for all 176 took 223.1 sec (3.17 sec model load). The complete scoring and 10,000-resample bootstrap finished in about 15 sec. No GPU compute or paid compute was used. The C: free-space reading after acquisition was 393,146,368 bytes. No model weights or legacy source artifacts were modified.

## Metrics

| Cohort/condition | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|
| New observer-disjoint validation, aligned images | 0.7955 | 0.5906 | 0.3029 | 0.0946 | 0.7622 |
| Macro over 8 taxa, aligned images | 0.7885 | 0.6001 | 0.3116 | 0.1826 | 0.7646 |
| Zero image features control | 0.1023 | 2.0824 | 0.8758 | 0.0330 | 0.1353 |
| Deranged image features control | 0.0795 | 6.1423 | 1.5025 | 0.6826 | 0.7622 |

| Taxon | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Rock Pigeon | 21 | 0.8571 | 0.3883 | 0.2101 | 0.1096 |
| Great Blue Heron | 19 | 0.7368 | 0.7425 | 0.3846 | 0.2529 |
| Mallard | 23 | 0.8696 | 0.4869 | 0.2139 | 0.1546 |
| Canada Goose | 18 | 0.9444 | 0.2919 | 0.1463 | 0.1188 |
| Northern Cardinal | 12 | 0.6667 | 0.8440 | 0.4648 | 0.3470 |
| American Robin | 24 | 0.5833 | 0.8656 | 0.5045 | 0.1806 |
| House Sparrow | 28 | 0.7143 | 0.7706 | 0.4225 | 0.2112 |
| Great Egret | 31 | 0.9355 | 0.4111 | 0.1459 | 0.0859 |

## Same-checkpoint comparison to the earlier validation

The same selected checkpoint scored 0.8155 accuracy / 0.5408 NLL / 0.2634 Brier / 0.0453 ECE on its earlier 683-record validation cohort (macro accuracy 0.8113, macro NLL 0.5584, macro ECE 0.0955). On the new observer-disjoint cohort, overall accuracy was 2.01 percentage points lower and NLL was 0.0498 higher; macro accuracy was 2.28 points lower, macro NLL 0.0417 higher, and macro ECE 0.0871 higher. This comparison is descriptive: the checkpoint was selected using the earlier validation cohort, and the two cohorts are not matched for difficulty or class count. It is consistent with useful observer-disjoint transfer while showing a calibration and modest performance gap; it is not an unbiased estimate of general deployment performance.
## Paired observer-cluster bootstrap

10,000 resamples, seed 20261029, 121 observer groups, percentile 95% intervals. Aligned minus zero-feature control: Accuracy +0.6932 (CI [0.6201, 0.7622]); NLL -1.4918 (CI [-1.6216, -1.3490]). Aligned minus deranged-feature control: Accuracy +0.7159 (CI [0.6444, 0.7789]); NLL -5.5517 (CI [-6.1719, -4.9426]). These are paired diagnostic intervals for this development cohort, not independent confirmation of generalization; the checkpoint was selected on the older validation data.

## Interpretation and limits

The selected representation/readout transfers substantially above the zero-feature and deranged-image controls to this new observer-disjoint cohort. This weakens the narrow explanation that the previous result depended only on observer overlap or image identity memorization. It does not resolve class-specific difficulty: Northern Cardinal and American Robin remain weakest, while sample counts per taxon are small (12–31). Calibration is imperfect and macro ECE (0.1826) exceeds the sample-weighted ECE (0.0946). The large difference from the older validation result must be interpreted cautiously because the cohorts may differ in acquisition, class composition, and difficulty; the old validation was involved in checkpoint selection. No further evaluation of this cohort is planned.

The frozen VideoPrism readout remains a narrow image specialist. This result does not establish temporal-video reasoning, AV fusion, reusable multimodal alignment, or achievement of the R0–R8 shared-Lens product gates. The new images were not added to training; the expansion-training gate stays closed.

## Artifacts

- Media: `C:\CodexArtifacts\inat-observer-transfer-media-20261029-attempt-01\`
- Feature vectors and row metadata: `C:\CodexArtifacts\inat-observer-transfer-eval-20261029-attempt-01\validation-features.npz`, `feature-rows.jsonl`, `feature-summary.json`
- Per-example probabilities and metrics/bootstrap: `C:\CodexArtifacts\inat-observer-transfer-eval-20261029-attempt-01\predictions.jsonl`, `evaluation-result.json`, and `score-evaluation-once.py` (SHA-256 `db42990b575b64268e8efaf844d955c2de43ce634f0f1789c72bcd8036627872`)
- Preflight and overlap audit: media directory `summary.json`, `overlap-audit.json`
- Pilot (first three sorted images): `C:\CodexArtifacts\inat-transfer-pilot-20261029-01\`

Existing Teacher/readout/checkpoints and 2026-10-26 train/validation media/features were treated read-only. No sealed audit, training run, GPU, cloud compute, or product promotion was used.
