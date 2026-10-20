# HoloAssist AVS window-exposure stratification — 2026-10-21

## Purpose and scope

This is a CPU-only, post-hoc analysis of already saved development-validation
predictions. It asks whether the AV-versus-AV+sensor comparison changes between
complete 30-second windows and shorter final windows. It performs no inference,
loads no model weights, selects no checkpoint, and does not read official
validation, test, or sealed-audit data. Existing training runs, datasets, and
checkpoints remain unchanged.

The same validation split was already used to select these checkpoints and for
prior diagnostics. This analysis is exploratory and selection-biased; its
bootstrap intervals describe session-resampling uncertainty within this split,
not independent generalization uncertainty.

## Inputs and reproducibility

Source branch/commit: `codex/r2-holoassist-results-20261020` at
`5c0bae773f9d4ec983ff3ac58fc13b1dd24bc47a`.

The frozen validation windows file is
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/temporal-windows-v2/validation.jsonl`
(SHA-256 `584f6e9dfc5b5590f5108d954064f3ad05745f77f5e4211c65a1e0f9aaeaf149`).
The prediction files and hashes are:

| Arm | Saved checkpoint predictions | SHA-256 |
|---|---|---|
| AV | `.../avs-readout-attempt-02/av/predictions-step-000960.json` | `1cba4c9a2bdc86385e7f380095dd9f4c47febb1a9d04732a24d0dd70bb291a4b` |
| Sensor only | `.../avs-readout-attempt-02/sensor_only/predictions-step-000256.json` | `b48bbc0eceb0b723276bb433fe517e8db0f899fa168db3a351d445b7e8c6ce40` |
| AVS | `.../avs-readout-attempt-02/avs/predictions-step-000256.json` | `910cc9c0551a184b2c9a188531f5ac4c64db7cc9f5a479f33c638c1ae38dedee` |

The audit requires exact equality between each prediction ID set and the
2,373 labeled validation IDs, and checks the session and target for every ID.
It excludes 125 unlabeled context windows. The labeled ID order hash is
`aa8e4a07f4e65229407481bd28bfeb404f5d71cc95141f9d736ef0df66b2c066`.
“Full” means duration equals 30 seconds within 1e-6 seconds; every shorter
labeled window is “tail.” Metrics are Accuracy, balanced Accuracy, NLL,
multiclass Brier for the binary distribution, 15-bin ECE, and mean confidence.
The paired AVS-minus-AV percentile intervals resample whole sessions 10,000
times (seed 1701 for full windows, 1702 for tails).

Command used:

```powershell
python -m scripts.analyze_holoassist_window_exposure `
  --windows C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\temporal-windows-v2\validation.jsonl `
  --prediction av=C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\avs-readout-attempt-02\av\predictions-step-000960.json `
  --prediction sensor_only=C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\avs-readout-attempt-02\sensor_only\predictions-step-000256.json `
  --prediction avs=C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\avs-readout-attempt-02\avs\predictions-step-000256.json `
  --full-window-seconds 30 --replicates 10000 --seed 1701 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\holoassist-window-exposure-audit-attempt-01\report.json
```

The machine-readable result is stored outside Git at the output path above;
its SHA-256 is
`964186EF21F02430F72F3C200FEDAE4AF16576ED02761462D46CE26B8BAF9449`.

## Results

| Stratum | n / sessions | Positive labels | Arm | Accuracy | Balanced Acc. | NLL | Brier | ECE-15 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Full 30s | 2,150 / 264 | 688 (32.00%) | AV | 0.7344 | 0.6670 | 0.5466 | 0.3633 | 0.0290 |
| Full 30s | 2,150 / 264 | 688 (32.00%) | Sensor only | 0.6823 | 0.5310 | 0.6183 | 0.4262 | 0.0393 |
| Full 30s | 2,150 / 264 | 688 (32.00%) | AVS | 0.7242 | 0.6479 | 0.5680 | 0.3812 | 0.0211 |
| Tail | 223 / 223 | 31 (13.90%) | AV | 0.8386 | 0.5952 | 0.4169 | 0.2535 | 0.0486 |
| Tail | 223 / 223 | 31 (13.90%) | Sensor only | 0.8565 | 0.5515 | 0.3616 | 0.2191 | 0.0336 |
| Tail | 223 / 223 | 31 (13.90%) | AVS | 0.8610 | 0.6217 | 0.3737 | 0.2255 | 0.0480 |

AVS-minus-AV paired session-bootstrap results:

| Stratum | Metric delta | Point difference | 95% session-bootstrap interval |
|---|---|---:|---:|
| Full 30s | Accuracy | -0.0102 | [-0.0234, +0.0028] |
| Full 30s | Balanced Accuracy | -0.0191 | [-0.0360, -0.0026] |
| Full 30s | NLL | +0.0214 | [+0.0081, +0.0349] |
| Full 30s | Brier | +0.0178 | [+0.0075, +0.0285] |
| Full 30s | ECE-15 | -0.0078 | [-0.0267, +0.0186] |
| Tail | Accuracy | +0.0224 | [-0.0045, +0.0538] |
| Tail | Balanced Accuracy | +0.0265 | [0.0000, +0.0680] |
| Tail | NLL | -0.0431 | [-0.0833, -0.0057] |
| Tail | Brier | -0.0280 | [-0.0549, -0.0038] |
| Tail | ECE-15 | -0.0006 | [-0.0492, +0.0424] |

## Interpretation

On full windows, AVS has lower Accuracy and balanced Accuracy than AV, and
worse NLL and Brier; the session-bootstrap intervals for balanced Accuracy,
NLL, and Brier exclude zero in the unfavorable direction. On tails, AVS has
better point Accuracy, NLL, and Brier, but only 31 positive examples are
present; the Accuracy interval includes zero. These strata differ in both
exposure duration and label prevalence, so this does not isolate a causal
effect of duration or establish sensor complementarity. Lower tail prevalence
and window duration are known corpus/task confounds (see the linked coverage
audit).

The result supports designing a prospective, fixed-duration task with
session-level isolation and a fresh development split before another fusion
claim. It does not justify more tuning on these exposed IDs, change any
checkpoint choice, or support promoting AVS.

## Verification

The analysis is covered by CPU tests for full/tail assignment, exclusion of
unlabeled context, and rejection of ID mismatches. No weights, media, or
checkpoint files are included in Git.
