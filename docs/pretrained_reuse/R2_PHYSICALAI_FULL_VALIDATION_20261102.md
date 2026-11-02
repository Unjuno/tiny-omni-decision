# PhysicalAI fixed-pilot full development validation — 2026-11-02

## Purpose and scope

This is a one-time, fixed-checkpoint extension of the 64-episode action pilot
reported in [the pilot record](R2_PHYSICALAI_ACTION_PILOT_20261102.md). The
unchanged pilot probes were scored on all 227 episodes assigned to the project
development-validation split (908 queries; four per episode). This is exposed
development data, not a blind test, sealed audit, or product-quality result.
The action codebook, probe weights, feature normalization, and evaluation
preprocessing were not refit or selected on these 227 episodes. No reserved
episode media was downloaded or evaluated. No model training was performed.

This run was motivated by the pilot's 64-episode validation limitation. It does
not resolve the source's missing scene/layout identifiers, establish scene
independence, or admit the source for R2-M. In particular, no training-coverage
expansion is authorized by these results alone; the existing source/task gates
remain in force.

## Frozen inputs and artifacts

- Source manifest: `manifests/pretrained_reuse/physicalai-singlearm-cabinet-left-split-20261102.json`, SHA-256
  `52a845a2c2033347b105806b8458848156c7b0a9a7e7c6489fc02a10543dab90`.
- Source revision: NVIDIA PhysicalAI Robotics Manipulation SingleArm,
  `73c00c18e7ca108e91846294b20a1d7cd5de9340`, `panda-open-cabinet-left`.
- Feature model: DINOv2-base, `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- Probe checkpoint: the original 64-train-episode pilot checkpoint, SHA-256
  `102965afe03de2cb6cb549e9e37df7355ea9063b605a086ee00079c3eb7c1da0`.
- Full-validation features: 227 episodes / 908 queries, SHA-256
  `3165d64ece23da134de9fbac59ac60216939da2c86821b622c5ba53583b9257f`.
- Predictions: SHA-256
  `103958268b9283f8faa5ccae0ae6d099c3f5df8ec04381c266635c8c658e0995`.
- Final result JSON: SHA-256
  `6985057d9703c3ebe6cd6da03456d333089c83544dfd0c525733ada0bd0878b4`.
- Artifact directory: `C:/CodexArtifacts/pretrained-reuse-r2/physicalai-action-full-validation-01/`.
- The result record states `fit_checkpoint_changed=false`,
  `checkpoint_selection_on_this_full_validation=false`,
  `reserved_holdout_payload_used=false`, and `validation_is_blind=false`.

The 227 validation videos and matching Parquet files totaled 117,453,770 bytes
and passed pinned publisher size/SHA-256 checks. The receipt records 454
verified files and 134.467 seconds wall time with eight download workers.
Feature extraction used CUDA FP32, batch size 1; it decoded 1,816 selected
frame appearances across 908 queries. Metadata records 59 seconds of decode
time and 1,132.697 seconds of model inference time. The inference timer includes
periods of contention from an unrelated GPU process; those timings should not
be interpreted as an uncontended throughput benchmark. Peak CUDA allocation
was not captured in the extraction metadata.

## Results

All rows use the same 908 validation queries and the fixed pilot checkpoint.
The train row is an in-sample diagnostic on the original 256 pilot training
queries; it is not an independent generalization estimate.

| Fixed probe/control | Accuracy | Macro accuracy | NLL | Brier | ECE (10 bins) | n |
|---|---:|---:|---:|---:|---:|---:|
| Proprioception history | 0.8095 | 0.7956 | 3.2787 | 0.3748 | 0.0227 | 908 |
| Video, mean of 8 frames | 0.7819 | 0.7464 | 1.6650 | 0.3891 | 0.1762 | 908 |
| Video, last frame | 0.7885 | 0.7380 | 1.4905 | 0.3742 | 0.1678 | 908 |
| Joint video + proprioception | 0.8139 | 0.8215 | 1.5054 | 0.3455 | 0.1618 | 908 |
| Joint, video replaced by train mean | 0.6013 | 0.5941 | 1.6115 | 0.6076 | 0.2209 | 908 |
| Joint, proprioception replaced by train mean | 0.7346 | 0.6675 | 1.8430 | 0.4653 | 0.2066 | 908 |
| Joint, episode-misaligned video | 0.5055 | 0.4687 | 5.8501 | 0.9323 | 0.3402 | 908 |
| Train class prior | 0.2192 | 0.1250 | 1.9437 | 0.8446 | 0.0113 | 908 |
| Pilot train, proprioception (in-sample) | 1.0000 | 1.0000 | 0.0000004 | 0.0000000 | 0.0000004 | 256 |
| Pilot train, video mean (in-sample) | 1.0000 | 1.0000 | 0.0000005 | 0.0000000 | 0.0000006 | 256 |
| Pilot train, last frame (in-sample) | 1.0000 | 1.0000 | 0.0000006 | 0.0000000 | 0.0000006 | 256 |
| Pilot train, joint (in-sample) | 1.0000 | 1.0000 | approximately 0 | approximately 0 | approximately 0 | 256 |

The episode-cluster paired bootstrap used 2,000 resamples, seed 17, sampling
227 episode IDs with replacement. Joint minus proprioception accuracy was
`+0.391` percentage points (95% interval `[-2.423, +3.304]`), which includes
zero. Joint minus 8-frame video was `+3.178` points (interval `[+1.211,
+5.286]`). The control contrasts were: joint minus video-missing `+21.264`
points (`[+17.952, +24.670]`); joint minus proprioception-missing `+7.919`
points (`[+5.617, +10.463]`); joint minus episode-misaligned video `+30.803`
points (`[+26.982, +34.692]`). These intervals are conditional on the fitted
pilot and this exposed validation set; they are not an independent audit.

## Interpretation and decision

The original 64-episode probes fit their training queries perfectly but reach
only 78.2–81.4% accuracy on the full development-validation set. This is a
large in-sample/generalization gap and is consistent with overfitting of the
small pilot. The joint probe reaches 81.4% accuracy and 82.1% macro accuracy,
but does not show a reliable accuracy gain over proprioception alone in the
episode-level paired interval. Its lower NLL/Brier than proprioception is
offset by high ECE, and neither result establishes calibrated decision quality.
Video interventions substantially change predictions and scores, showing
that the fitted joint probe uses the video representation on these examples;
they do not prove that video adds repeatable information beyond proprioception
for an independently defined scene/task distribution.

**Disposition remains `REVIEW_NOT_ADMITTED`.** Do not claim generalized
multimodal improvement, scene-independent generalization, or progress toward
the 95% target from this action-prototype task. Before any larger fit, resolve
the task/scene identity and label stability gates in the source and pilot
records. No checkpoint, codebook, or model configuration is selected or
promoted by this evaluation.

## Reproduction and environment

The evaluation command was:

```powershell
python scripts/evaluate_physicalai_full_validation.py `
  --pilot-artifact-dir C:/CodexArtifacts/pretrained-reuse-r2/physicalai-action-pilot-01 `
  --validation-artifact-dir C:/CodexArtifacts/pretrained-reuse-r2/physicalai-action-full-validation-01 `
  --bootstrap-replicates 2000
```

The feature extraction ran locally with the pinned DINOv2 model in FP32 on the
RTX 3080 Laptop GPU. No cloud compute or training was used. The full validation
features and per-example probabilities are retained outside Git. The reserved
227-episode payload remains untouched. Because this development validation has
now been inspected, it must not be described as blind or used as a fresh
checkpoint-selection set.
