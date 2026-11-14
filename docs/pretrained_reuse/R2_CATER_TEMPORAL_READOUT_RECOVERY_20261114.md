# R2 CATER temporal readout recovery (2026-11-14)

## Scope and disposition

This is a development-only, question-conditioned video Decision probe using
frozen VideoPrism-B interval features. It does not establish text-free video,
general video understanding, multimodal quality, or product readiness. The
fixed validation split has already participated in readout selection and is
not blind evidence. No sealed audit was opened.

The older 2026-11-13 source-screen entries still say `REVIEW_NOT_ADMITTED`
because the CATER/Clevr-4 visual-identity check was then pending. The later
bounded check in `R2_CATER_TEMPORAL_PATH_A_PREFLIGHT_20261113.md` completed at
02:23 JST on 2026-11-14, before the frozen-feature run directory was created at
07:23 JST. It compared 67,480 selected CATER interval frames with 10,531
previously-used non-sealed Clevr-4 images, found zero exact RGB64 matches, and
manually reviewed all 174 global nearest-pHash pairs at distance <=2 and the
48 closest same-suffix pairs. No duplicate was identified; the receipt records
the bounded scope and residual limitation (SHA-256
`e4c906fb758f8678cb1e8bc239e36d92962f9a9c99e660b01bfeaa18f5d83499`). Together
with the local non-redistributive research license interpretation in
`R2_CATER_LICENSE_SCOPE_FOLLOWUP_20261113.md`, this cleared the stated local
development gate for this fixed candidate. It does not admit CATER for M-2,
product-quality claims, or sealed-audit use.

The first readout attempt completed both conditions and wrote their selected
head checkpoints, then failed while generating the `permuted_video` control:
it mapped whole videos without checking that a replacement had the same task
frame interval. The failure was a post-fit evaluation bug, not a training
failure. The original attempt directory and checkpoints were retained
unchanged. The failed process did not produce its final report or prediction
files; its epoch history and exact selected epochs are therefore unavailable.
The source code configured at most 12 epochs with patience 4 and selected by
macro-over-interval validation NLL, but the actual epoch counts cannot be
recovered from the saved heads alone.

The recovery adds deterministic interval-compatible video mapping and a
checkpoint-only evaluation path. It reloaded the two saved heads, copied them
byte-for-byte to a new output directory, and recomputed train/validation
predictions and controls. It did not run optimizer steps or modify source
checkpoints/features/manifests.

## Reproduction

- Repository: `Unjuno/tiny-omni-decision`
- Source commit: `073b050112668ceed13b1fa2844af65652fff995`
- Branch: `codex/cater-permuted-control-fix-20261114`
- Invocation:

```powershell
python scripts/train_cater_temporal_readout.py `
  --manifest-dir D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-decision-manifest-20261113-attempt-03 `
  --feature-dir D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-interval-features-20261114-attempt-01 `
  --text-model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --output-dir D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-readout-20261114-attempt-02 `
  --seed 17 --cpu-threads 4 `
  --reuse-checkpoints-from D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-readout-20261114-attempt-01
```

- Evaluation-only run time: 2026-11-14 09:38:32–09:58:04 JST (19m31s).
- Device: CPU, 4 configured Torch threads. No GPU training or optimizer update.
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.19.0,
  NumPy 2.4.6. CUDA was available in the environment, but this run explicitly
  executed on CPU.
- Frozen encoders: VideoPrism-B, 114,365,184 parameters; MiniLM, 17,389,824.
  Readout: 510,721 trainable parameters. Total unquantized resident count:
  132,265,729. No quantized package size was measured.
- Cohort: 4,360 training queries / 1,120 validation queries; 308 validation
  video groups. The validation option count varies from 5 to 10 (not a binary
  task).
- Target: select the object with greatest annotated net displacement over
  the supplied 75-frame interval.
- Train manifest SHA-256:
  `23ccef3c1eebf90bafb6726d1933c8105b0acf96f6c02d581d50bc95c56de2e8`
- Validation manifest SHA-256:
  `dd9e7920026228d1e8442aea45883c36f1def7f54e96dd126856165b20079fa0`
- Frozen feature result SHA-256:
  `1de4384a2aa164a61f9036857ba74c56af7d0ebdad55d1582fa3d783361ed917`
- Recovery script SHA-256 at execution:
  `4b0cd928d33805f8e12ff01f49f707c967ba4867577f2c59feb96a28791e4744`

The two source checkpoints were copied without transformation. Source and
recovery SHA-256 values match:

| Checkpoint | SHA-256 |
|---|---|
| `best-video-readout.pt` | `a3776b87154f8172faec463cca95cfc61d9e25877ff43f84b0c496beb6750e0b` |
| `best-text_only-readout.pt` | `44b3fda44947216fff9afe1a904249aab59348b792df357f72e0633f6f9b3df0` |

## Validation metrics

Metrics below are macro-averaged across the four 75-frame question intervals;
each row also covers all 1,120 validation examples.

| Condition | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Video + question | 0.1886 | 1.8817 | 0.8371 | 0.0306 |
| Question-only | 0.1950 | 1.8812 | 0.8370 | 0.0329 |
| Static first frame | 0.1886 | 1.8818 | 0.8371 | 0.0303 |
| Shuffled interval interior | 0.1886 | 1.8817 | 0.8371 | 0.0306 |
| Reversed interval | 0.1886 | 1.8818 | 0.8371 | 0.0307 |
| Different video, same interval | 0.1886 | 1.8817 | 0.8371 | 0.0302 |

Video minus question-only macro Accuracy is -0.0064. The paired video-group
bootstrap 95% percentile interval is [-0.0207, +0.0077] (2,000 resamples,
seed 1701). This interval includes zero. Input-destruction and video-swap
controls change the scores negligibly, so this run provides no evidence that
the fitted readout uses the video features for this task. The nearly unchanged
question-only score is a warning that this task/setup may not identify useful
video dependence; it does not prove the frozen representation has no video
signal.

## Artifacts and integrity

Recovery output:
`D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-readout-20261114-attempt-02`

- `report.json` SHA-256:
  `eaf1f1b0f34396e2dae048565753e21f7c705f6f377d685286fef1a17722ca66`
- `video-validation-predictions.jsonl` SHA-256:
  `d97f26e6f23c18ea95f1206e9666dafce28997441b0bfaea9b399da6fd2de765`
- `text_only-validation-predictions.jsonl` SHA-256:
  `53e00dcb7049ed74eb909a10bb314d9eab36b59b512d707a7fcbf9c422e96aa3`

The two prediction files each contain 1,120 unique validation IDs, identical
paired ID sets, finite nonnegative probabilities summing to 1, and consistent
target/prediction/correct fields. Original attempt-01 checkpoints, feature
cache, manifests, and other project worktrees were left unchanged.

Focused CPU tests: 9 passed. Ruff: passed. This is a successful recovery of the
evaluation run, but the quality result is weak and does not pass an R2
multimodal-dependence gate. Do not tune repeatedly against this validation set
or promote this readout. Choose the next R2 task from the existing admitted
task matrix and fresh-group evidence.
