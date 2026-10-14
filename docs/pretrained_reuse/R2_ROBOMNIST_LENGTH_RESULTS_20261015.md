# R2 — RoboMNIST source-length shortcut measurement

**Executed, complete directory-only diagnostic. Not native modality quality.**
The source's camera file lengths alone predict60-class activity at65.45% and
robot identity at85.30% on the fixed development split. Keep this control in
future native-input comparisons; do not count these results toward the Omni
95–98% target. Compression length can reflect genuine physical activity as well
as capture/codec artifacts; this experiment alone does not establish their cause
or disqualify the source.

## Prospective protocol and actual run

[Protocol](R2_ROBOMNIST_LENGTH_PROTOCOL_20261015.md) committed before any fitting
at4524d55; reviewed artifact/failure/resource corrections committed at
`d171a4f8877adb2e8ef781ae7e5b122d836358fd`, the actual producer source commit.
No estimator, split, seed or numerical rule was tuned after observing metrics.

```powershell
python -m scripts.diagnose_robomnist_lengths --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-length-diagnostic-01
python -m scripts.replay_robomnist_lengths --main C:/CodexArtifacts/pretrained-reuse-r2/robomnist-length-diagnostic-01 --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-length-replay-02.json
```

PID45324 ran normally from2026-10-14T16:00:44.972848+00:00 to
2026-10-14T16:00:45.658401+00:00 (2026-10-15 01:00:44–01:00:45 JST), exit0.
Actual body0.685534s, peak CPU64,503,808B, complete output3,400,531B.
The bounds were120s,2GiB CPU,16MiB output,1GiB disk reserve. CPU NumPyfloat64,
Python3.11.9,NumPy2.4.6; effective environment, CLI arguments and80 source/code/
library/protocol pins are retained. No dependency upgrade or cloud cost.

All2,135 groups/all60 classes are retained. Repetition numbers1–32 occur in all
classes;33–44 have incomplete class coverage. Group all classes/views with the
same numeric repetition, rather than assuming each class's Rp is independent.
Rp01 remains train. SHA256 ordering with seed17 gives validation blocks
**2,6,8,9,10,12,14,18,21,22,29**:660 groups, exactly11/class. Train has1,475 groups,
21–33/class. All seven input views of each group stay together. No source group
or numeric Rp crosses the proposed development split; actual payload-content
overlap and true collection-session independence remain unverified.

These records have now been used for a supervised metadata diagnostic. Neither
this validation nor a posthoc repartition of this experiment is a new blind
audit. Native-input development may use this exposed source with that limitation;
external or otherwise genuinely untouched evidence is still needed for blind/OOD
quality claims. No existing sealed audit was opened.

## Fixed predictors and estimator

Only seven log1p input file sizes are predictors. Filenames/class/robot/speed/
repetition/uncertainty IDs, CRCs, archive sizes and positionGT sizes are excluded.
Directory identities bind labels and splits in a separate ledger. No media/GT
payload is opened by this diagnostic. It is not a frozen pretrained encoder,
new Teacher or deployed Decision model.

Four fixed nearest-centroid arms use train-only mean/std and60 train centroids;
logits=-0.5 standardized squared distance, temperature1, equal class priors.
The train-count prior is the fifth no-input control. There are no epochs,
checkpoint selection, calibration fitting, hyperparameter search or extra seeds.
All train/validation60-class log probabilities are saved. Robot2,velocity3 and
trajectory10 posterior tasks are exact marginals, not simplified alternatives
used to redefine the60-class result. Four questions share each recording.

## Validation results — all five conditions

All rows below use the same660 source groups. Per-condition full Accuracy/NLL/
Brier/ECE/train gap and paired intervals are in the
[20-row CSV](R2_ROBOMNIST_LENGTH_METRICS_20261015.csv) and
[fresh replay JSON](R2_ROBOMNIST_LENGTH_RESULTS_20261015.json).

| Diagnostic predictors | Activity60 Acc | Robot2 Acc | Velocity3 Acc | Trajectory10 Acc | Activity60 NLL / Brier / ECE |
|---|---:|---:|---:|---:|---|
| Camera file lengths |65.45%|85.30%|58.48%|32.73%|2.9359 / .9084 / .5821|
| Microphone file lengths |1.97%|51.36%|36.67%|11.21%|4.1635 / .9859 / .0129|
| CSI JSON length |22.27%|63.79%|48.33%|16.52%|3.5229 / .9567 / .1893|
| All seven lengths |43.48%|83.18%|71.82%|34.55%|2.4645 / .8424 / .2993|
| Train-frequency prior |1.67%|50.00%|33.33%|10.00%|4.0971 / .9834 / .0057|

Camera length activity60 train/validation Accuracy64.88%/65.45%; all-length
train/validation53.56%/43.48%. Combining fields improves velocity and NLL but
decreases full activity Accuracy relative to camera lengths. No best-arm adoption
or product selection is made. Audio length activity60's paired difference from
prior includes0; its velocity marginal has a small positive interval, so do not
describe every microphone-length task as strictly chance.

Paired complete-Rp percentile bootstrap:2,000 draws,seed17,11 validation blocks,
resampling all records from each selected block. Camera length activity60 minus
prior=+63.79pp,95% interval[+58.48,+67.73]pp; microphone length=[−0.76,+1.36]pp;
CSI=[+18.79,+22.42]pp; all-length=[+39.09,+44.85]pp. These describe this exposed
development source, not independent laboratory/session population uncertainty.
NLL uses stable logs with no floor; Brier is full multiclass squared error,
ECE is15-bin top-label error. The large camera-length ECE shows poor calibration.

## Verification, provenance and failure protection

Fresh-process replay reconstructs all source rows/features/splits, recomputes
every arm and2,000-draw interval, and verifies all **7 arrays /640,500 full-class
log probabilities byte-exactly**. An additional independent scalar implementation
checks40 train/validation metric groups (six fields each) at absolute1e-12.
The estimator implementation is shared/pinned; the scalar metric arithmetic is
independent. Reconstructed source/split/order/class accounting and all84 source/
artifact bindings pass. Fresh replay time/peak are recorded in the machine JSON.
Its consumer was uncommitted at execution; exact consumer SHA is retained there.
Replay01 is preserved. Review identified late main-report hashing and missing
terminal product rehash in that first replay. RED→GREEN mutation regressions
now bind the exact main-report bytes before parsing and terminal-recheck all84
source/product/report pins. Fresh replay02 passes; numerical results are unchanged.

Pre-execution review found late config hashing, no canonical saved-array readback,
missing incomplete terminal records and a resource snapshot taken before final
readback. All were corrected before the real fit: artifacts are hashed from
prepared bytes at creation, config/rows/all arrays are read back, failures preserve
`FAIL_INCOMPLETE_LENGTH_DIAGNOSTIC`, and terminal time/peak/disk/output guards run
after the last readback. RED→GREEN regression tests cover these cases. The only
actual experiment here finished normally; simulated test failures are not
unsuccessful alternative experiments. No estimator/split changes were made.

| Artifact | SHA-256 |
|---|---|
| Main result | `05f62ad3b67328a7164a966ba0b8198c03cd41b4e4745e8a0e16d719520a93ea` |
| Effective config | `25a2926a856fa31a5a7a5275720b8bac7728fc74a4d4fc5bfadb5e73b60eacc8` |
| Source rows | `2584656bb2c917200dce6cf5cb4d8a5d12dfc655fcdb8a087b32ff8df29749ad` |
| Arrays | `f8d4ed11c79faee9643a378545bc94d4bca29a48dbdf39b51b8d5b46a97cd262` |
| Group order | `4a447e59b5a7c1957767cbbe5de97bfb7f7adaa468ff691e0adf7f446f0dbb9a` |
| Development split assignments | `0be2a8776029e05c1a53066dfc4d3a719ab783ede0def06b7ac6318cc5ae947b` |

NPZ/source ledgers remain outside Git. Only code/tests/protocol and small metrics/
reports are published. Prior raw-input and stopped native-memory artifacts remain
intact; no GPU job restart, legacy training intervention, sealed evaluation,
ternary quantization, Recovery, promotion or main merge.

Fresh verification: Ruff passes; CPU pytest1,484passed/1skipped (20.37s),
NoTorch1,070passed/96skipped (6.12s); all existing candidate/model manifests,
four catalogs and the policy audit pass. Read-only whole-publication review finds
no remaining Critical/Important issue. All84 current diagnostic pins,92 earlier
raw-input pins and50 protected native-memory pins remain unchanged.

## Next action

Keep length-only and no-input controls in future native-input development.
Verify invariance to container metadata/codec changes on a known train pilot;
define fixed, justified decoded-input preprocessing that excludes source file
length/identity features. Episode-level tasks must explicitly limit temporal
claims, because common-clock and audio capture/export evidence is still missing.
Only then admit a bounded frozen-feature comparison with proper source/session/
content checks and identical cohorts. No native fit was started in this stage.
