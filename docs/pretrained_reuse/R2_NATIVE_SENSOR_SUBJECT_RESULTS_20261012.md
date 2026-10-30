# Native sensor Decision: subject-disjoint HARTH results

## Status

The pre-registered sensor-only comparison is already complete; this report
records and audits the preserved run rather than repeating training. The
selected models are development results on three held-out participants. They do
not meet the Omni sensor-quality goal, establish broad sensor understanding,
or qualify the optional sensor path for product use.

## Frozen protocol and data boundary

The comparison follows
[`R2M_SENSOR_SUBJECT_PROTOCOL_20261012.md`](R2M_SENSOR_SUBJECT_PROTOCOL_20261012.md):
seed 17; all 12 activity codes; 128-sample windows, stride 64, no interpolation,
gaps over 30 ms split windows, mixed-label windows retained, center-sample
target; float32 AdamW (`lr=0.001`, `weight_decay=0.01`), batch 128, 12 epochs,
no class weighting or early stopping. The temporal Conv1d readout is 6,700
parameters. Its static mean/std/min/max baseline is 2,380 parameters and is
permutation-invariant; it is not capacity matched.

The frozen prepared corpus contains 14 train subjects and 3 validation subjects
(S013, S019, S027), with 62,269 train windows and 12,763 validation windows.
The two output prediction files use identical validation IDs and targets. The
external prepared-corpus receipt records archive SHA-256
`2ab54d6b11467ddbe998dfb60355f9f1ae181b0af7a89e7c78bb7af2cf16b64a`, split
SHA-256 `0259363d0dde5255a7c2ba2925959aad12df78007c60d26db855b009fca04695`,
and manifest SHA-256
`f22d57e70f6e812ecb1156c3823b93ee27198e8c2969007d6346985bdd6bc9d2`. The
manifest contains only train and validation subject rows; its receipt states
`calibration_or_audit_opened: false`. The later read-only diagnostic iterated
those same 17 manifest rows, not other subject files. Exact sample-value overlap
between train and validation was zero.

No calibration or sealed-audit result is available or claimed. Validation
checkpoint selection used the same three participants later used for reported
development metrics, so those metrics are selection-affected and not blind.
The three-cluster paired interval is highly discrete and low-powered.

## Results

Checkpoint selection was earliest minimum validation NLL. Each run retained
separate best and final state files; a best-state reload reproduced the saved
probabilities exactly.

| Readout | Best epoch | Parameters | Accuracy | Balanced accuracy | NLL | Brier | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal Conv1d, mean pool | 9 | 6,700 | 0.9147 | 0.7326 | 0.2412 | 0.1185 | 0.0201 |
| Static mean/std/min/max | 12 | 2,380 | 0.9387 | 0.6980 | 0.1894 | 0.0873 | 0.0116 |
| Temporal Conv1d, center readout (separate follow-up) | 11 | 6,700 | 0.8925 | 0.5953 | 0.3100 | 0.1466 | 0.0239 |

The static baseline has higher sample Accuracy and better NLL/Brier/ECE, while
the mean-pooled temporal model has higher balanced Accuracy. The precomputed
exhaustive bootstrap over the three validation-subject clusters (27 draws)
favored static on Accuracy, NLL and Brier point estimates. Its 95% intervals for
temporal-minus-static were `[-0.04684, -0.00062]` Accuracy,
`[0.00305, 0.09063]` NLL, and `[0.00216, 0.05801]` Brier. With only three
clusters and validation-based selection, this is weak development evidence,
not a reliable population comparison.

The task is strongly imbalanced: class index 6 contributes 6,604/12,763
validation windows (51.7%). The temporal model has only 15.4% recall on class
index 2 (733 windows); static has 49.9%. Classes 9 and 11 have zero validation
support and class 10 has six examples, so these classes cannot be assessed
reliably. On mixed-transition windows, temporal/static Accuracy is 51.5%/58.9%
versus 95.5%/97.4% on unmixed windows. These results expose a weak transition
case and why sample Accuracy alone overstates quality.

Shuffling the 128 acceleration samples within each validation window, while
retaining timestamps and labels, changes the selected mean-pooled temporal
model from `91.47% / 0.2412` to `84.64% / 0.4054` Accuracy/NLL. Static-model
outputs are unchanged to floating-point precision. This supports temporal-order
sensitivity for this narrow HARTH task; the permutation creates unrealistic
signals and is not a general causal test. The selected temporal training curve
continued to improve through epoch 11 and regressed at epoch 12; static NLL
continued to its best value at epoch 12. No claim of a settled capacity ceiling
follows.

## Runtime and preserved artifacts

- Primary temporal/static run source commit: `15f8e60984f2bca93e239bb7b3c77d577ddd70a7`.
- Separate center-readout follow-up source commit: `71d6eb0290f20739c37a16686b5949a66dca303b`.
- The primary config records PyTorch `2.6.0+cu124`, `device: cuda`, and 34.30 s
  total measured run time. Per-model elapsed times are 21.54 s (temporal) and
  11.68 s (static); peak process VRAM recorded was 341,386,240 bytes. GPU model,
  driver, Python version, and system-wide concurrent GPU load were not captured
  in the run config and remain UNKNOWN. The center follow-up records 21.76 s and
  the same peak allocation. Cloud cost was $0.
- Primary config SHA-256: `eddaef150b4e9500dd482472c806085b767882916ba79856f45ebbbe38794d83`.
- Primary result SHA-256: `ee56119dfa4503690be903e7b090264043188dc69c82670aab6e596a0d590aac`.
- Temporal best/final SHA-256:
  `74c887f665d5637b985b6b839cbcddc53acedb9a75be1bf35c1232092afe2ace` /
  `35bceeb216b6aba6b5947e3913772c5a69202e1ba202c4174cc7d600744e0b13`.
- Static best/final SHA-256:
  `5780b04c5022de2f6a085c27032de05c5f86e84a6c638b02a93ed91a9b7db40d` /
  `a758f36f55802452e108546536b34da8b6ed4e54fdadd19c179ca387f90d289f`.
- Subject diagnostic receipt SHA-256:
  `86e93d4cfa3b275b676d073f9f4eb99ee91083718fe547aaab8e4596d58205cd`.
- External outputs remain under `C:\CodexArtifacts\pretrained-reuse-r2\`:
  `sensor-subject-training-01`, `sensor-center-training-01`, and
  `sensor-subject-diagnostics-01`. They are not tracked in Git.

The diagnostics summarize only the train/validation predictions and exact
train/validation feature-value overlap. The held-out-subject count, imbalance,
mixed-window failure and selection on the same validation are the main
limitations. Do not tune more on these three subjects; obtain a new, untouched
subject cohort or keep this result as a narrow engineering baseline. It does
not justify adding xLSTM or a large pretrained sensor tower to the full Omni
model.
