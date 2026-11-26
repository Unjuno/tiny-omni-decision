# Frozen joint video/text matching diagnostic

## Scope and result

Completed all 99 training videos / 198 beginning/end questions, with normal
video and black-image controls: 396 saved prediction rows. This is an observed
training-side diagnostic, not held-out or sealed evaluation. No fine-tuning,
temperature fitting, prompt search, checkpoint selection or seed selection.

| Input | Accuracy | NLL | Brier | ECE | Both conditions correct |
|---|---:|---:|---:|---:|---:|
| Normal video | 0.252525 | 1.123952 | 0.683650 | 0.130653 | 0.111111 |
| Black frames | 0.252525 | 1.213399 | 0.751624 | 0.248287 | 0.121212 |

Normal initial/final Accuracy: 0.242424 / 0.262626. Only one of 99 videos changes
the predicted option when the question changes from beginning to end. A uniform
three-option distribution has NLL 1.098612 and Brier 0.666667, both better than
the normal video's measured values. This ITM-to-choice readout is not adopted.
Among the 53 videos whose correct option changes, both-condition correctness is
0/53 for both normal and black inputs.

Video-cluster paired bootstrap: 10,000 draws, seed17, percentile95. Normal minus
black Accuracy interval [-0.055556, 0.055556]; NLL [-0.129769, -0.046733];
Brier [-0.097251, -0.037130]. The probability improvement over black input does
not establish correct temporal reasoning or scene-specific evidence use.

## Reproduction

- Backbone: All-in-one Tiny, source `c6f4d5d18109f99ea748b11ed26a4b9239212cf4`.
- Checkpoint SHA256 `d79415e85a6ade9d14e932bead53fadd7c61248fcd72808e2c6d86d46e2e9ca7`.
- Strict-loaded matching path: 11,430,914 resident parameters; MLM-only weights excluded.
- Train pairs SHA256 `8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7`.
- Media manifest SHA256 `469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3`.
- Pinned BERT vocabulary admitted by the repository loader; compact statements
  preserve the supplied candidate text. No input truncation.
- Three frames: first/middle/last before the exclusive cut (full clip for -1).
  Resize256/random-crop224, normalization0.5, seed17 reset before each clip's
  transforms and each candidate's inference. Upstream random patch selection retained.
- Score: ITM positive-minus-negative logit, softmax over the three options.
- CPU FP32, four Torch threads; Python3.11.9, Torch2.6.0+cu124,
  Transformers5.6.2, PyAV18.1. No cloud compute or GPU run.
- Loop wall time567.679 seconds; excludes initial model load and one-video setup.
  Peak process memory was not measured; do not infer it from parameter count.

Artifact directory:
`C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-train-diagnostic-02/`.
Includes protocol, provenance, predictions, completion and report JSON.
Predictions SHA256 `7b20f3629089ad206e2d95ad53667361241893f0555a51cf5a8aabfb9fbedc0c`.
Runner: `all-in-one-source-initial-01/train_diagnostic_black.py`, SHA256
`a7f6f04293c88159d11a88277d660fe2054ed9d1b803e532f9b25c72aaea65f9`.
Runtime source state86cc58f; reporter source5409a52.

```powershell
python scripts/report_joint_video_diagnostic.py --run C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-train-diagnostic-02 --pairs C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-pairs-01/train.json --output C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-train-diagnostic-02/report-replay.json
```

## Limitations and disposition

Attempt01 failed because normalized zero pixels are upstream padding, producing
zero-size positional interpolation. Preserve its partial outputs as incomplete.
Attempt02 uses valid normalized black pixels (-1), with no source/model change.

Three sparse frames can miss the occlusion event; random crops can remove objects,
and the pretrained matching objective is not this viewpoint/ordinal task.
Independent crops and stochastic patch ordering are preserved audit choices,
not proof of an optimal temporal pipeline. Black frames are an out-of-distribution
control; a positive NLL difference is insufficient to demonstrate scene semantics.
These results do not establish a model-family ceiling or justify more fitting.

Product checkpoint licensing remains REVIEW. Do not promote, redistribute weights,
or proceed to compression from this result. Existing v1/v2 and other artifacts
remain untouched. The broad Omni quality and fusion gates remain unmet.
