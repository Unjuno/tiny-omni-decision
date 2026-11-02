# AVATAR native spatial localization: lower-learning-rate follow-up (2026-11-02)

## Purpose and limits

This is a one-factor, CPU-only follow-up to the frozen-feature TRAIN-only
localization probe. The first joint readout was severely overconfident
(validation NLL 5.603, ECE 0.481) and selected its first epoch. This follow-up
tests whether reducing the AdamW learning rate alone improves that readout.
It is development-validation evidence for a 40-cell AVATAR sound-source
localization task, not general multimodal quality or product-model evidence.

The same train and development-validation IDs, row order, DINOv2 and
EfficientAT features, feature normalization, seed, model definitions, optimizer,
weight decay, batch size, epoch cap, early stopping, and validation selector
were used. The only training setting changed was learning rate: `0.02` to
`0.002`. Video-only and audio-only arms were rerun at both rates as controls.
The existing `readout-01` outputs were not overwritten. Official validation
and sealed audit were not accessed.

## Reproducibility record

- Repository: `Unjuno/tiny-omni-decision`
- Work branch: `codex/r2-avatar-native-localization-20261102`
- Source commit before this follow-up change: `c16cd78ba9dc9500ce8bcc2211ccd48ce553a64d`
- Modified fit script SHA-256: `0b65ba4a8fdbb283d45d49eed2f2ac83f3a5f68d354e8feee298faa056868e01`
- Effective-config SHA-256: `222e6e55f57fafb1b9eb9c092af54865f62eed0de673e330df2985f2178ffc65`
- Train JSONL SHA-256: `1dd445a9faf8eed3e35b36cf9aa06a921ebe2629f8084885c0fe658a9a1265b1`
- Development-validation JSONL SHA-256: `a565a863fd30560bde26b75660d3743b943ddba4055593d4ab130a27db020b40`
- DINOv2 spatial feature SHA-256: `df25d3a18f5873e2927cd5b733102fc017bc9e62170802a29db087b7b540b278`
- EfficientAT feature SHA-256: `4ba7316edf719e595a7f5a6a659ae9535e06f5441416968bcc305360744fd540`
- Train/validation: 9,693 / 2,311 rows, 1,817 / 454 source-video groups,
  zero source-group or media-identity overlap. Audio is missing for 38 / 11
  rows and represented by zero content plus the existing missingness flag.
- Command:

```powershell
python scripts/fit_avatar_native_localization.py `
  --train-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-native-localization-20261102-attempt-01\fold\train.jsonl `
  --validation-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-native-localization-20261102-attempt-01\fold\validation.jsonl `
  --video-features C:\CodexArtifacts\pretrained-reuse-r2\avatar-native-localization-20261102-attempt-01\spatial-features.npz `
  --audio-features C:\CodexArtifacts\pretrained-reuse-r2\avatar-dinov2-train-fold-20261102-attempt-01\audio-features-train-only.npz `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-native-localization-20261102-attempt-01\readout-lr002-01 `
  --learning-rate 0.002
```

- Machine: local Windows CPU process; Python 3.11.9, NumPy 2.4.6,
  PyTorch 2.6.0+cu124. CUDA was not used. Fit time was 17.489 seconds;
  cloud cost was zero. The RTX 3080 Laptop GPU remained at 0% utilization
  during the recorded post-run check.
- Artifacts are external to Git under
  `C:\CodexArtifacts\pretrained-reuse-r2\avatar-native-localization-20261102-attempt-01\readout-lr002-01\`.
  The 15 files total 8,437,850 bytes. They include effective config, full
  histories, selected and final weights, per-example predictions, the paired
  comparison receipt, and reload verification. Checkpoint files are not in Git.

## Results

The validation selector was unchanged: target-set NLL, `min_delta=0.001`,
patience 10, maximum 100 epochs. The table reports the selected checkpoint for
each arm. Accuracy is hit@1 over the annotated target cell set.

| Learning rate | Readout | Selected epoch / epochs run | Accuracy | NLL | Brier | ECE | Mean confidence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.020 | Video-only | 6 / 16 | 45.39% | 1.751 | 0.726 | 0.140 | 0.594 |
| 0.002 | Video-only | 5 / 15 | 47.43% | 1.602 | 0.690 | 0.088 | 0.532 |
| 0.020 | Audio-only | 1 / 11 | 21.72% | 3.059 | 0.983 | 0.243 | 0.460 |
| 0.002 | Audio-only | 12 / 22 | 24.75% | 2.474 | 0.876 | 0.118 | 0.365 |
| 0.020 | Joint AV | 1 / 11 | 39.94% | 5.603 | 1.044 | 0.481 | 0.880 |
| 0.002 | Joint AV | 1 / 11 | 48.77% | 1.513 | 0.654 | 0.064 | 0.550 |

For the joint readout, lowering the learning rate changed validation Accuracy
by +8.83 percentage points, NLL by -4.090, Brier by -0.390, and ECE by -0.417.
For Video-only the corresponding changes were +2.03 points, -0.148, -0.036,
and -0.052. These are paired comparisons on the same exposed development
validation rows. A 2,000-resample, seed-17 source-group paired bootstrap gave
the joint Accuracy change a 95% percentile interval of [+5.93,+11.67]
percentage points; the candidate-minus-reference NLL interval was
[-4.549,-3.639]. Because both checkpoint selection and comparison used this
same validation partition, these intervals are descriptive and are not
independent confirmation.

At learning rate 0.002, joint minus Video-only Accuracy was +1.34 points, but
its paired source-group bootstrap 95% interval was [-1.55,+4.07] points. The
joint readout is audio-sensitive: shuffling available audio lowered Accuracy
to 32.24% and raised NLL to 2.222; zeroing audio content while preserving the
missingness flag yielded 38.99% and NLL 1.799. This supports dependence on
audio content, but does not establish a reliable joint-quality gain over the
Video-only model.

Overfitting is reduced but not absent. The selected joint checkpoint is still
epoch 1 of 11: train NLL 0.976 versus validation NLL 1.513, with an Accuracy
gap of 18.28 points. The 22.07% smoothed fit-set location prior remains a
weak baseline; the Audio-only result is only 24.75% hit@1. The new result
therefore supports a learning-rate sensitivity in this particular interaction
readout, not the claim that audio is generally useful for localization or that
the AVATAR task is solved.

## Save/reload check and disposition

In a fresh Python process, `video_only-best.pt` and `joint-best.pt` were loaded
with `weights_only=True` and `strict=True`, then evaluated on the same 2,311
validation examples. IDs and order matched; both argmax arrays matched exactly;
maximum absolute probability difference against the saved predictions was
`2.3842e-7`. The receipt is `reload-verification.json` (SHA-256
`9e124d81a3ca973ade1cfb4a2dbbe72ea850851300956371041ac669015b3db1`).

Keep the lower-rate joint readout as a useful validation baseline for the next
diagnostic, but do not promote it as a product model. Retain Video-only as the
best supported localization baseline until a new, task-appropriate independent
evaluation or further predeclared evidence establishes joint benefit. The
frozen encoder package remains above the provisional 200 MB target, and this
task does not address text, general image understanding, speech, or broad
video reasoning. R2 remains open.
