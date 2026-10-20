# HoloAssist IMU coverage and window exposure audit — 2026-10-20

## Purpose and limits

This CPU-only audit checks whether zeros in the fixed IMU coverage features
represent missing sensor samples or the zero padding used to place partial
final windows into 30 bins. It also compares positive-label prevalence in
final and full windows. The audit reads the existing frozen train and
development-validation artifacts only. It did not load sealed-audit data,
run a model, select a checkpoint, alter any dataset, or start training.

The development-validation split has already been used for model selection and
diagnostics. Its findings below are exploratory, not independent evidence.

## Data and method

The existing builder calculates 1-second IMU bins, concatenates 9-axis means,
9-axis standard deviations, and three sensor coverage values, then places each
window into a 30-bin array. `pool_temporal_features()` zero-fills bins beyond
the window end. The audit derives the valid-bin mask from the frozen window
start/end times and verifies the feature row IDs match the JSONL IDs in order.

The audit script is `scripts/audit_holoassist_sensor_coverage.py`. Its output
is stored outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/coverage-audit-attempt-01/coverage-audit.json`.
The output states `sealed_audit_loaded: false`. Input SHA-256 values are
recorded in that JSON:

- train windows: `406b4e6cf18588efb931809ac74415609b6b17500dd1ac02225df48ca9514b24`
- validation windows: `584f6e9dfc5b5590f5108d954064f3ad05745f77f5e4211c65a1e0f9aaeaf149`
- IMU train features: `25e4d45a332df418e1934ee0a5e5d2a1ee72ef0ec7dfb1a3d884250ad1f628b5`
- IMU validation features: `2a3b79e6ffb72815e5943ad2820d395a5ae874832d32f3417b2009db06ea082d`
- IMU receipt: `5520afc81e28e1ee8f9675b284d23202b21b9cfc9b8daab8f61de32bb2118e41`

## Findings

| Split | Zero coverage bins | Explained by padding | Zero bins in valid observations | Final-window positive rate | Full-window positive rate |
|---|---:|---:|---:|---:|---:|
| Train | 14,543 | 99.732% | 39 / 278,386 valid bins | 12.26% (873 windows) | 30.02% (8,478 windows) |
| Development validation | 3,709 | 99.757% | 9 / 71,240 valid bins | 13.90% (223 windows) | 32.00% (2,150 windows) |

All three sensor coverage channels are exactly identical in the stored
features. Every padding bin has zero coverage. The remaining valid zero
coverage bins are rare (0.0140% of valid train bins and 0.0126% of valid
validation bins) and are consistent with genuinely unobserved 1-second bins.

The final windows average 17.67 seconds in train and 17.36 seconds in
validation; each has about 11.8–12.1 padded bins. They also contain fewer
actions on average (6.14 train / 5.91 validation) than the full windows
(9.84 / 9.97). Since the target is positive when **any** action in the window
is incorrect, shorter windows naturally have less opportunity to receive a
positive label. Thus the coverage vector exposes window duration/final-window
status, which correlates with label prevalence. This is a data/task-design
confound, not proof of leakage or the sole cause of the AVS result.

## Interpretation and follow-up

The sensor intervention results remain as previously recorded. This audit
shows the current coverage channels mostly encode padding/duration, rather
than fine-grained sensor availability. It does not invalidate the frozen run,
but it limits the interpretation of a learned sensor gate and any claim that
sensor observations add task-relevant evidence.

Do not revise these labels, alter the existing split, or retrain using the
already exposed validation set. A future sensor study should predefine
fixed-duration windows or include duration/exposure explicitly in the target
definition, keep session-level split isolation, and assess on a fresh
development split before any sealed evaluation. Preserve the current run and
its artifacts unchanged as the historical baseline.

## Reproduction

```powershell
python -m scripts.audit_holoassist_sensor_coverage `
  --windows-dir C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\temporal-windows-v2 `
  --imu-dir C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\imu-features-attempt-01 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\holoassist-textfree-unused-train-recovery-02\coverage-audit-attempt-01\coverage-audit.json
```

Three CPU unit tests cover padding-vs-valid zero accounting, sample-ID order
rejection, and coverage-channel consistency. No weights or media were added to
Git.
