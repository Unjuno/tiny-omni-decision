# EgoViz fixed-model transfer to the prior pilot

The [protocol](R2_EGOVIZ_PILOT_TRANSFER_PROTOCOL_20261030.md) and original
runner were committed as `1aa4a820` before evaluation. The existing 24-session
checkpoint and its fixed training-out-of-fold inverse temperature **3.56727**
were evaluated without fitting on the prior pilot's 16 sessions. The pilot and
new 44-session corpus had **zero overlap** in session, chunk and sample IDs.

| Prior pilot split | Model and scaling | Accuracy | NLL | Brier | ECE | Confidence |
|---|---|---:|---:|---:|---:|---:|
| Train, 106 examples / 8 sessions | New checkpoint, unscaled | 65.09% | 1.1595 | 0.6293 | 0.3016 | 0.3494 |
| Train | New checkpoint, fixed scalar | 65.09% | 0.9273 | 0.5023 | 0.1042 | 0.5805 |
| Development, 110 examples / 8 sessions | New checkpoint, unscaled | 50.91% | 1.2449 | 0.6774 | 0.1816 | 0.3275 |
| Development | New checkpoint, fixed scalar | 50.91% | 1.1304 | 0.6152 | 0.0855 | 0.5141 |
| Development | Previous pilot checkpoint, unscaled | 47.27% | 1.2773 | 0.6919 | 0.1501 | 0.3226 |

The new checkpoint did not train or select on these 16 sessions. Both pilot
splits were used previously in developing the task and architecture, and the
old checkpoint was selected on the pilot development set. Therefore this is a
useful transfer check for the new weights, **not a blind architectural or
product evaluation**. The old checkpoint's development metrics reproduced the
previously saved report within `1e-6`. Its pilot training result is excluded
from this table because that checkpoint trained on those records.

On pilot development, the new model's Accuracy difference versus the old
checkpoint was **+3.64 percentage points**, with paired eight-session bootstrap
95% interval **[-10.71, +18.93]**. Its unscaled NLL difference (new minus old)
was **-0.0324** [-0.0993, +0.0257]. These intervals include zero and do not
establish a clear model improvement.

Applying the fixed scalar on pilot development improved observed NLL by 0.1145
and Brier by 0.0621; paired eight-session 95% intervals were **[-0.0177,
+0.2525]** and **[-0.0043, +0.1333]**, respectively. Both include zero.
On the other eight pilot sessions, observed NLL improved by 0.2322 [0.1655,
0.3025] and Brier by 0.1270 [0.0881, 0.1660]. All intervals use 2,000 paired
session resamples and seed 17, holding model and scalar fixed; the small number
of session groups and prior use of the data limit their interpretation. Window
samples overlap in time within a session, and participant/site independence
across sessions is unknown. Scaling preserved every option ranking.

The pilot's MOMENT features were computed with CUDA and the new 44-session
features on CPU. The encoder checkpoint and source revisions matched, but
backend numerical equivalence has not been measured. This possible difference
is not evidence that backend choice caused the result. No action labels,
sealed media, new training, development-fitted scalar, product promotion or
quantization were used. The 50.91% result remains far below the product goal
and covers timestamp correspondence, not general decision semantics.

## Reproducibility and correction

The initial run was successful, but its training-split bootstrap records called
the resampling unit `development session` because a shared helper used that
label. The actual resampled groups were the correct **pilot train sessions**;
no samples or numbers were wrong. A reporting-only correction was committed as
`6c8aa6ac`, and a fresh process repeated the same fixed inference in a new
directory. All 216 per-example predictions had an identical SHA-256 and all
metrics and intervals were exactly equal except for the corrected unit label.
Both original and corrected outputs are retained.

The [machine-readable result](../../manifests/pretrained_reuse/egoviz-pilot-transfer-results-20261030.json)
is the corrected fresh-process result. Artifacts:

- Original output: `C:\CodexArtifacts\egoviz-pilot-transfer-20261030-attempt-01`.
- Fresh-process output: `C:\CodexArtifacts\egoviz-pilot-transfer-20261030-reload-01`.
- Predictions SHA-256: `dc287a3f10ae253c79eb809c19ebe1b00cc67b6c55d86bd155c24f517d982149`.
- Initial results SHA-256: `0fb1d4a5a7acbfa4cab9c75f5440c861e9db757c51e01cbabd0f07b7ae935acf`.
- Corrected results SHA-256: `17d375bd9fe2152b539831a87807c40b250735f2f9608516c1ad702f8bcebbb8b`.
- Reload receipt SHA-256: `69bdf1da7ca592acd9506aee820157e45c5f4a7fcd7ac60b5594cfc1819c88c0`.
- New model checkpoint SHA-256: `b62c1cac0c38c459263dbc2c5ffd8eddc76b8654a2254a5f4d15fac8fe95a862`.
- Fixed scalar metadata SHA-256: `179cae04c90acf59a2dac0146538b2d8f48a69907136a650b75c773361a8ede8`.

Both runs used Python 3.11.9, PyTorch 2.6.0+cu124 and NumPy 2.4.6 on CPU.
Each took approximately seven seconds wall time. The original command was:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$py = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
$new = 'C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02'
& $py -m scripts.evaluate_egoviz_pilot_transfer --pilot 'C:\CodexArtifacts\egoviz-vs-sync-20261029-attempt-01' --current "$new\full-aggregate" --selected "$new\readout-results-cpu-01" --calibration 'C:\CodexArtifacts\egoviz-train-oof-temperature-20261030-attempt-01\results.json' --output 'C:\CodexArtifacts\egoviz-pilot-transfer-20261030-attempt-01'
```

Reproduction must use a new output path. The observed calibration improvement
is encouraging but uncertain on this small previously observed pilot. Further
work should improve ranking on condition-sensitive decisions and verify both
ranking and calibration on appropriate independent data before promotion.
