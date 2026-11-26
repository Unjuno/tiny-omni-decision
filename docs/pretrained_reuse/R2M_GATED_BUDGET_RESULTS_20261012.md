# Gated readout learning curve: budget helps NLL, then overfits

Separate protocol and runner frozen at **16c1da5**. One seed17 fresh-init run
completed1,000epochs, preserving the original100-epoch artifact. Exact best
reload passed. The first101 checkpoints (epochs0..100), all four arrays each
(train/validation probabilities and targets), reproduce the original run
**bitwise**. Effective config matches except declared epoch budget and source
provenance; row IDs/order are identical. No hidden input or optimizer change.

## Fixed endpoints and independently reported selector

| Epoch | Train NLL | Validation Accuracy | NLL | Brier | ECE |
|---:|---:|---:|---:|---:|---:|
| 100 | 1.606334 | 0.897436 | 1.729142 | 0.732459 | 0.706161 |
| 250 | 0.281290 | 0.820513 | 0.656572 | 0.274602 | 0.216903 |
| **395 selected** | 0.071996 | 0.820513 | **0.542434** | 0.237947 | 0.081886 |
| 500 | 0.038151 | 0.820513 | 0.557535 | 0.248595 | 0.118000 |
| 750 | 0.014480 | 0.820513 | 0.616141 | 0.269833 | 0.111802 |
| 1000 fixed end | 0.007698 | 0.820513 | 0.664919 | 0.281930 | 0.144371 |

The fixed1000-minus100 validation NLL difference is-1.064222, paired subject
cluster percentile95 interval[-1.545927,-0.627264],10,000draws,seed17. This is
single-seed developmental evidence that100epochs left probability quality
undertrained. Accuracy moves in the opposite direction (35/39 to32/39). Do not
describe it as uniform improvement or a newly achieved quality threshold.

After selected epoch395, train NLL falls0.0720->0.0077 while validation NLL
rises0.5424->0.6649 and Brier worsens. **Overfitting is now observed** under this
fixed architecture/corpus. This supports ending this budget experiment, not
an inference that all possible architectures or data budgets have plateaued.
No further epochs or seeds were launched.

## Same selected checkpoint controls

| Condition | Accuracy | NLL |
|---|---:|---:|
| Matched | 0.820513 | 0.542434 |
| Shuffled video | 0.307692 | 3.506616 |
| Video absent | 0.641026 | 1.172234 |
| Shuffled sensor | 0.512821 | 1.912297 |
| Sensor absent | 0.666667 | 0.864257 |

Both inputs affect predictions; this is not sufficient for useful fusion.
Sensor-only remains1.0Accuracy/0.163704NLL; scaled linear fusion0.871795/0.350965.
Selected-minus-sensor NLL difference+0.378730, interval[0.040544,0.623871].
Selected-minus-scaled-linear+0.191468, interval[-0.051550,0.513970].
The candidate is **not adopted**. Missingness remains outside its training
distribution, and the same fixed permutation/float32 tolerances apply.

## Coverage, compute and reproducibility

130unique training episodes, not130,000 new examples. Full run consumes130,000
episode encounters (1,000full-batch updates); selected395 consumes51,350.
Same421,826 trainable parameters;114,787,010 with frozen VideoPrism for this
probe, not a full Omni resident ledger. CPU train/evaluate/save loop88.699s,
including writing1,001 prediction snapshots and cumulative history. No GPU or
paid compute, no encoder updates, new data, augmentation or post-hoc calibration.
Process peak working set605,655,040bytes; frozen encoders were not resident
during this cached-feature readout run.

Executed in a separate Python process using `runpy.run_module` on
`scripts.train_vidimu_gated_budget`, then saved
`psutil.Process().memory_info()._asdict()` to process-memory.json. Normal module
invocation reproduces the same training path:

```text
python -m scripts.train_vidimu_gated_budget
python -m scripts.verify_vidimu_gated_budget
python -m scripts.compare_vidimu_gated_budget
```

All directories under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `vidimu-gated-fusion-budget-01`: exact config/source/input hashes, best/final,
  normalization, records, full history and all predictions, process memory.
- `vidimu-gated-budget-verification-01`:404-array prefix replay, config/record
  comparison, fixed-endpoint interval and learning-curve accounting.
- `vidimu-gated-budget-controls-01`: matched/shuffle/absence predictions,
  selected-checkpoint intervals and numerical option-permutation checks.

Best SHA256:`21a3e6066a3b141e74d5468348fec953725e09ce7da3ad814ab0408159aedc57`.
Final SHA256:`9d0850f7d4f5afbdcefba2c571cd397362b5001e44a5a27ed1c954a19b7f7d2c`.

Only3 validation subjects and repeatedly observed development data; bootstrap
does not remove checkpoint-selection bias. No sealed audit was read. Prior
negative results remain intact. Main Omni quality and complementary temporal
multimodal tasks remain open; do not replace them with this narrow13-class score.
