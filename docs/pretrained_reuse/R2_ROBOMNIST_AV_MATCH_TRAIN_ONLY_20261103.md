# RoboMNIST AV-match train-only stage — 2026-11-03

## Status

The fixed seed-17 readout fit is complete; the full fixed validation evaluation
is pending. This is an execution-order adjustment to the preregistered
candidate, not a result or a quality claim. The original feature cache remains
`INCOMPLETE` at 1,778 / 1,780 groups. The only absent groups are the two frozen
validation groups listed below.

The train-only stage read **only** the 780 train groups (1,560 binary rows).
It did not open a validation feature, produce a validation prediction, or
select a checkpoint using validation. The 400-update fit and all three readout
definitions match the predeclared schedule. Once the original cache reaches
1,780 groups, evaluation must load this exact checkpoint and must not refit.

## Fixed inputs and missing validation features

- Source commit at run time: `2ab435342dd2640801ee3f9aebc076b78e03e1a5`.
- Frozen manifest SHA-256:
  `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Original feature cache:
  `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-cache-20261103-attempt-01`.
- Verified training feature groups: **780 / 780**.
- Missing groups, both validation-only:
  `Ar1_Vhigh_Ac1_Rp10_UNC085` and `Ar1_Vhigh_Ac1_Rp22_UNC304`.
- The frozen validation remains **1,000 groups / 2,000 rows**. No partial
  validation metric was computed.
- The original cache `run.json` remains `INCOMPLETE`; its extractor and 2 GiB
  reserve were not changed. This stage wrote to a separate directory.

## Training and artifact

Command:

```powershell
python scripts/fit_robomnist_av_match_train_only.py `
  --manifest C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-split-20261103-attempt-03\manifest.json `
  --cache C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-cache-20261103-attempt-01 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-readout-20261103-attempt-02-train-only
```

The fixed configuration was seed 17, rank 32 for joint AV, 400 full-batch
updates over all 1,560 train rows, AdamW, learning rate `0.01`, weight decay
`0.0001`, and gradient clipping at norm 5. The audio-only and video-only
logistic controls each received the same 400 updates.

| Readout | Trainable parameters | Train CE, step 1 → 400 | Fit seconds |
|---|---:|---:|---:|
| Joint AV | 73,728 | 1.05886 → 0.00215 | 6.61 |
| Audio only | 769 | 0.82041 → 0.69323 | 0.99 |
| Video only | 1,537 | 0.78955 → 0.69378 | 1.59 |

These are training losses only. They do not establish validation accuracy,
cross-modal benefit, generalization, or product quality.

- Start: `2026-11-03 23:27:16 JST`; finish: `23:27:25 JST`.
- Total wall time recorded by the fit: 9.30 seconds.
- Runtime: Python 3.11.9, NumPy 2.4.6, PyTorch 2.6.0+cu124, CPU; CUDA was not
  initialized and no GPU was used.
- Save/reload parity: exact train-logit match for all three readouts.
- Output directory contains a 306,962-byte checkpoint; SHA-256:
  `3592bdd0cd8343f600de1654daa21d1b920392a41f8c3c71725d3f90dff4f00d`.
- Effective config SHA-256:
  `f15f42ce644800fc6fd237cc36d3b30e5b00c8c579ba1451ebb464cc38fe9c5c`.
- Train feature hash:
  `9293348d83a77d2b0788531cd4127e294b90d723f733a520d3fb13f0b5895df1`.
- Normalizer and implementation hashes are preserved in the external
  `run.json` and `config.json` under the output directory.

The standard environment used by this fit (Python 3.11.9 / NumPy 2.4.6)
differs from the feature-extraction runtime recorded by the cache (Python
3.11.16 / NumPy 1.26.4). The later frozen evaluator verifies the train-feature
and normalizer hashes and will fail closed if they differ; do not silently
recompute or replace the checkpoint if that happens.

## Execution safeguards and next step

Before training, three C: samples over 20 seconds averaged 1,379,580,587 bytes
(1.285 GiB), with a 6.72 MiB spread. The unrelated Quartus process was absent
and the RTX 3080 was idle. The train-only writer required at least 1 GiB free
and created a separate sub-megabyte artifact. It did not bypass the feature
extractor's 2 GiB reserve because it performed no extraction. The missing
validation data still cannot be extracted until that original reserve is
restored.

Focused checks after implementation: Ruff passed; 10 related CPU tests passed.
The frozen-fit loader accepted the saved checkpoint and verified the run,
config, checkpoint, implementation, manifest, train-feature, and normalizer
hashes. The evaluator's complete-cache path has not yet run end to end.

Next, when stable C: free space is at least 2 GiB, resume the **original** cache
with the existing extractor and `--resume`. Then evaluate all 2,000 validation
rows once through `fit_robomnist_av_match_readout.py --frozen-fit
<train-only-output>`, using a new evaluation output directory. The evaluator
must reuse these weights without fitting and keep the full validation
denominator. The two missing validation source groups have prior metadata-only
cross-task exposure documented in
`R2_ROBOMNIST_CROSS_TASK_METADATA_EXPOSURE_20261103.md`; keep that limitation in
the eventual result. Do not open sealed audit data.
