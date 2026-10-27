# Dense DROID train windows: matched-history sensor ablation

The [fixed protocol](R2_DROID_DENSE_HISTORY_SENSOR_CV_PROTOCOL_20261027.md)
trained two CPU linear heads for 50 epochs per fold on all 80,998 prebuilt
train-recording windows. The five held folds remained disjoint by the original
275 recording identities; every held window was evaluated once after training.
Both heads receive previous command; the second also gets the same 112-value
past sensor representation. A seven-by-seven transition table fit on each
outer training fold is a separately specified history-only reference.

| Predictor | Accuracy | Macro accuracy | NLL | Macro NLL | Brier | ECE-10 | Mean recording accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| History-only head | 0.90812 | 0.90145 | 0.46311 | 0.50372 | 0.17380 | 0.02329 | 0.91087 |
| History plus sensor head | 0.90722 | 0.90044 | 0.44993 | 0.49308 | 0.17533 | 0.02178 | 0.91022 |
| History-only transition table | 0.90812 | 0.90145 | 0.45301 | 0.49452 | 0.17313 | 0.00254 | 0.91087 |

The fixed primary difference, sensor-head minus history-only-head macro NLL,
was **-0.01064**. Sensor input modestly reduced NLL but did not improve
Accuracy or Brier, and its macro NLL is close to the different transition
reference. The learned heads have 56 versus 840 trainable parameters; this is
not an equal-capacity comparison. No hyperparameter, epoch or checkpoint was
selected from held predictions.

On 73,556 stable-command windows, learned history-only and history-plus-sensor
Accuracy were 100.00% and 99.895% respectively. On **7,442 changed-command
windows**, they were **0 / 7,442** and **4 / 7,442** (0.054%); the transition
table was 0 / 7,442. Changed-window NLL was 3.8332, 3.6781 and 3.9751
respectively. These are target-defined descriptive subsets; all windows remain
in the primary metrics. Dominant-axis changes may include small fluctuations.

Mean training loss across the five folds at epochs 1 / 10 / 20 / 30 / 40 / 50
was 1.865 / 0.910 / 0.578 / 0.495 / 0.471 / 0.462 for history-only and
1.599 / 0.728 / 0.516 / 0.459 / 0.440 / 0.431 for history-plus-sensor. These
are rolling training losses, not a fixed-example train/validation gap. Training
loss was still falling at 50 epochs; a held learning curve was not measured,
so neither a plateau nor an absence of overfitting is established.

All windows in the same recording share a fold, but repeated windows remain
correlated and operator/site independence is unverified. The cohort was
previously inspected, the label prevalence is highly persistent, and none of
this establishes general video, audio, fusion or product-quality performance.
No validation/audit records, video, GPU or product checkpoint were used. CPU
execution used PyTorch 2.6.0+cu124 with four threads and took 19.53 seconds.
Independent replay of all ten saved heads verified fold-training-only means
and scales, parameter counts and held probabilities; maximum absolute
difference was `5.96e-7` using NumPy versus saved PyTorch predictions.

Private output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-dense-history-sensor-cv-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`dc9375007f3dc759bc944b29c0e6c17bf6469431d31bdd18728cf2df2aa04c2b`.
The receipt records all 50 training losses per fold, natural-prevalence and
change/stable metrics, group counts and hashes for ten heads and OOF
probabilities. The original dense windows remain unchanged.
