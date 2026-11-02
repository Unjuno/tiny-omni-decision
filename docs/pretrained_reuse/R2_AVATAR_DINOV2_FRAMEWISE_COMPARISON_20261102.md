# AVATAR train-only DINOv2 framewise readout comparison — 2026-11-02

## Result

The frozen, framewise DINOv2-base representation did **not** improve the
selected joint AVATAR readout over the existing VideoPrism feature cache on a
new source-group validation fold derived only from AVATAR TRAIN. Both joint
readouts selected the same 91.23% sample accuracy. DINOv2 had worse NLL and
Brier, while its ECE was lower. The video-only accuracy difference was small
and uncertain. This is a narrow category-match screen, not evidence that the
product model meets its quality target.

The comparison is not a backbone-only ablation: DINOv2 encodes one frame at
the annotation timestamp, while the existing VideoPrism representation pools
eight frames over a one-second window. It can screen the practical single-frame
substitution, but cannot separate backbone quality from temporal-context loss.
No follow-up DINOv2 run is justified by these results alone.

## Data and split integrity

- Source: AVATAR revision `a7f0146ff704799d044133e61137836cacc1c233`, using only
  the pinned cross-source category-match TRAIN records.
- The official AVATAR validation split and sealed audit were not loaded.
- Seed 17 assigned underlying source groups by deterministic SHA-256 rank to
  fit and development-validation pools (80/20). Candidate choices were rebuilt
  from each pool independently; the query and all four candidate source groups
  in a row stay within the same pool.
- Fit: 9,630 choices / 1,814 source groups / 9,633 unique media assets.
- Development validation: 2,325 choices / 454 total source groups / 2,362
  unique media assets. The paired bootstrap resampled the 448 audio-query
  source groups represented in these rows.
- Train/validation source-group, media-asset, and query overlap were all zero.
- The new four-choice task still asks which cross-source video category matches
  an audio query. It is not same-scene localization or the native AVATAR task.
- The shared frozen EfficientAT audio cache has 647 rows whose decoded audio
  padding differs from the declared annotation window (maximum 3,248 samples).
  This pre-existing discrepancy is identical in both arms and remains a data
  limitation.

The fold manifest is at
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\fold-manifest.json`
(SHA-256 `7211c333bc4eb50a75995a8964f99c2cfdb39719b3731eb7b2c050d9d9a346c0`).
The ordered validation ID hash is
`59e50d72c2c4065d690483467d4185f189a1e56f6bd637091a17ce7f9caefbc8`.

## Pinned representation and runtime

| Component | Revision / properties |
|---|---|
| DINOv2 | `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`; Apache-2.0; 86,580,480 parameters; checkpoint SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841` |
| Existing video comparison | VideoPrism-B revision `be719a406d563b66f0ac969e7c94bab8e997c81a`; 114,365,184 parameters; checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` |
| Audio input in both arms | EfficientAT DyMN-0.4 frozen 384-D features; 1,961,239 parameters; checkpoint SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` |
| Learned joint readout | 73,857 parameters; 64-unit MLP, AdamW; audio/video encoders frozen |

The DINOv2 encoder plus EfficientAT and the joint readout total 88,615,576
parameters. DINOv2's FP32 checkpoint alone is 346,345,912 bytes, above the
provisional 200 MB package target. No deployment package, quantization,
end-to-end raw-media runtime, or product promotion was produced.

Feature extraction decoded the existing `video.zip` in place; it did not
extract or retain media. The archive SHA-256 is
`5ce5be7fa6dda85889750f6c7cc2fad807527063e7d7090c664170a66b078a68`. A single
nearest frame was selected per candidate asset, and the maximum decoded-frame
time error was below `1.8e-15` seconds. The 11,123 by 768 feature cache has SHA-256
`e51beff9f056df8b51f2d71a46226c03f30030aee4149f4b7ee156e4ece95cdd`.

The RTX 3080 Laptop GPU (16 GB) extracted those features in 2,816.184 seconds;
peak CUDA allocated memory was 567,159,808 bytes. The HotGPU guard waited for
the existing Orbitune compute process to yield, launched this extraction, and
released its reservation afterward. No process was manually stopped. Both
readout fits ran on CPU in 18.0 seconds (VideoPrism) and 16.2 seconds (DINOv2).
Environment: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2,
PyAV 18.1.0, NumPy 2.4.6. No cloud compute was used.

AVATAR's dataset card declares CC BY 4.0; the original YouTube media have
separate rights. The media were not redistributed. No additional license was
accepted for this experiment.

## Training and selection

Both readouts used the same source-group fold, audio cache, seed 17, CPU
backend, model class, optimizer, and validation selector. The concat and
conditional-gate arms were fit separately from frozen features. The joint
selector was best validation NLL with `min_delta=0.001` and patience 6. It
selected epoch 6 of 12 for VideoPrism and epoch 5 of 11 for DINOv2. Fixed-train
to-validation NLL gaps were 0.1919 and 0.2036 respectively.

The measured joint result is sample-weighted over the same 2,325 choices:

| Video features | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| VideoPrism eight-frame window | 91.23% | 0.2149 | 0.1206 | 0.0253 | 0.9364 |
| DINOv2 single frame | 91.23% | 0.2279 | 0.1252 | 0.0183 | 0.9305 |
| DINOv2 minus VideoPrism | 0.00 pp | +0.0130 | +0.0046 | -0.0070 | -0.0059 |

For video-only scoring, VideoPrism reached 31.23% accuracy, 1.3587 NLL,
0.7397 Brier, and 0.0613 ECE; DINOv2 reached 31.66%, 1.3607, 0.7385, and
0.0506. The paired source-group bootstrap difference in accuracy was +0.43 pp
for DINOv2, with a 95% percentile interval from -1.68 to +2.59 pp.

The joint DINOv2-minus-VideoPrism accuracy difference was 0.00 pp, with a
source-group paired 95% interval from -1.31 to +1.29 pp. Both intervals used
2,000 resamples, seed 17, and the audio-query source group as the resampling
unit. Macro-over-category joint metrics were also worse for DINOv2: accuracy
89.50% vs 90.52%, NLL 0.2661 vs 0.2342, Brier 0.1451 vs 0.1324, and ECE 0.0934
vs 0.0859.

The concat readout's missing-audio and missing-video controls were both near
four-choice chance (DINOv2 25.98% and 25.38%, respectively). However, the
post-hoc audio-category oracle scores 97.55% by using hidden candidate-category
labels to map class probabilities to options; it is not deployable. The
ordinary audio category classifier reached 79.83% across 79 classes. These
results do not establish a formal both-input-required gate, nor do they show
that the constructed task measures general multimodal decision quality.

## Artifacts and reproducibility

Implementation was committed at `f54f1dcb06e3a4227b031a1c9228e2de6fb5a757`
(`Add train-only AVATAR DINOv2 fold pipeline`). The branch is
`codex/r2-avatar-dinov2-frame-readout-20261102`.

- Fold/caches/predictions and extracted feature metadata:
  `C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\`
- VideoPrism fit report and per-example predictions:
  `C:\CodexArtifacts\pretrained-reuse-r2\avatar-videoprism-trainfold-readout-20261102-attempt-01\`
- DINOv2 fit report and per-example predictions:
  `C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-trainfold-readout-20261102-attempt-01\`
- Paired comparison JSON SHA-256:
  `600a326ae6b1231320d8f448a3e61140c99ecd0e2d763390661934cb473ce821`

Large feature caches and predictions remain outside Git; no media or model
weights were added. Sealed audit access was false for every step.

The exact commands, from the repository root, were:

```powershell
py -3.11 scripts/build_avatar_train_only_group_fold.py --train-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\train-model-input.jsonl --train-sidecar C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\train-analysis-sidecar.jsonl --audio-cache C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\audio-features-subset.npz --videoprism-cache C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\video-features-subset.npz --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01

py -3.11 -m scripts.run_with_hotgpu_guard --max-wait-seconds 21600 -- C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe scripts/extract_avatar_dinov2_train_frames.py --train-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\train-input.jsonl --validation-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\validation-input.jsonl --video-zip C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261101-attempt-01\video.zip --model-dir C:\CodexArtifacts\pretrained-reuse-r2\dinov2-base-runtime-smoke-01\model --output C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\dinov2-features-train-fold.npz --device cuda --batch-size 16

py -3.11 scripts/fit_avatar_decision_readout.py --train-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\train-input.jsonl --validation-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\validation-input.jsonl --train-sidecar C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\train-sidecar.jsonl --validation-sidecar C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\validation-sidecar.jsonl --audio-features C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\audio-features-train-only.npz --video-features C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\videoprism-features-train-only.npz --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-videoprism-trainfold-readout-20261102-attempt-01 --device cpu

py -3.11 scripts/fit_avatar_decision_readout.py --train-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\train-input.jsonl --validation-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\validation-input.jsonl --train-sidecar C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\train-sidecar.jsonl --validation-sidecar C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\validation-sidecar.jsonl --audio-features C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\audio-features-train-only.npz --video-features C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\dinov2-features-train-fold.npz --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-trainfold-readout-20261102-attempt-01 --device cpu
```

## Decision and next work

Do not replace VideoPrism with single-frame DINOv2 on this evidence. The result
does not justify expanding the fusion gate or treating the 91% task score as a
product result. R2 remains open. The next valuable step is to define and verify
a task where the label requires information from both synchronized modalities,
then evaluate it on source/scene-disjoint groups. If another representation
comparison is run, use the same timestamp window and frame sampling for both
encoders so temporal input is controlled.
