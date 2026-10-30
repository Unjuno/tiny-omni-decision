# EgoViz training-session cross-fitted temperature — results

Five CPU readouts were fitted under the [fixed follow-up protocol](R2_EGOVIZ_OOF_TEMPERATURE_PROTOCOL_20261030.md),
committed as `59679229` before execution. Each training session was held out
once from both fitting and inner checkpoint selection. The resulting 332
out-of-fold scores, drawn from all 24 training sessions, fitted one inverse
temperature **3.56727** (temperature **0.28033**). It was then applied to the
original selected checkpoint on the same 307 development examples.

| Fixed checkpoint on development | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Original | 55.70% | 1.2190 | 0.6607 | 0.2235 | 0.3335 |
| Scalar fitted on original training predictions | 55.70% | 1.4737 | 0.6430 | 0.2189 | 0.7759 |
| Scalar fitted on training-session out-of-fold predictions | 55.70% | 1.0688 | 0.5663 | 0.0391 | 0.5312 |

The last scalar used training-session labels only. On development, original
minus scaled NLL was **+0.1502**, with paired 20-session bootstrap 95%
interval **[+0.1029, +0.1971]**; the Brier difference was **+0.0944**
[+0.0707, +0.1195]. Bootstrap uses 2,000 session resamples, seed 17, and holds
the fitted scalar fixed. Positive scaling preserved all option rankings.

The prior checkpoint was selected on this development set, and this protocol
was motivated by the already observed calibration diagnostic. These are
exploratory development results, not a blind measurement. The five auxiliary
models each fit only 14–15 of the 24 sessions, so their score distribution may
differ from the original model's. Capture sessions may share participants.
The out-of-fold training metrics after scaling fit the scalar on those same
332 labels and are not an independent calibration test. No scalar is promoted
and the sealed split remains unopened. The approach addresses confidence
scaling; the 55.70% ranking Accuracy and 95–98% product goal remain unmet.

| Fold | Fit / inner validation / held-out examples | Selected / stopped epoch | Fit time (s) |
|---:|---:|---:|---:|
| 0 | 191 / 74 / 67 | 20 / 32 | 1.60 |
| 1 | 191 / 67 / 74 | 12 / 24 | 1.10 |
| 2 | 196 / 69 / 67 | 19 / 31 | 1.89 |
| 3 | 208 / 55 / 69 | 17 / 29 | 1.63 |
| 4 | 210 / 67 / 55 | 17 / 29 | 1.47 |

Fresh-process reload verified each fold's source hash and disjoint groups,
regenerated all 332 out-of-fold logits, refitted the scalar, and reproduced
all 307 scaled development probabilities. Maximum differences were **0.0**
for logits, scalar and probabilities. Original weights and input hashes were
checked again. Python 3.11.9, PyTorch 2.6.0+cu124 and NumPy 2.4.6 executed on
CPU, with no paid compute or GPU training.

The complete fold assignments, per-epoch histories and metrics are in the
[machine-readable result](../../manifests/pretrained_reuse/egoviz-train-oof-temperature-results-20261030.json).
External artifacts are under
`C:\CodexArtifacts\egoviz-train-oof-temperature-20261030-attempt-01`:

- Five `fold-*-state.pt` checkpoints, retained separately.
- `training-oof.jsonl`: SHA-256 `75c37dec18497245b1c8d887798b5d17e44c1bb05ce859910ffa7764512ece0d`.
- `development-predictions.jsonl`: SHA-256 `c75064dd16bf0b7fff597c41eb6b709b4ba0d404a96846b6ac431df67813e576`.
- `results.json`: SHA-256 `179cae04c90acf59a2dac0146538b2d8f48a69907136a650b75c773361a8ede8`.
- `reload-verification.json`: SHA-256 `eac92b9b3a7923109f234f1dffe7292201efb70fd938037d33381ad41dc92a57`.

Exact original run and reload commands, from the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$py = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
$data = 'C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02'
$out = 'C:\CodexArtifacts\egoviz-train-oof-temperature-20261030-attempt-01'
& $py -m scripts.evaluate_egoviz_train_oof_temperature --corpus "$data\full-aggregate" --selected "$data\readout-results-cpu-01" --output $out
& $py -m scripts.verify_egoviz_train_oof_temperature --corpus "$data\full-aggregate" --selected "$data\readout-results-cpu-01" --result-dir $out --output "$out\reload-verification.json"
```

The existing output is protected from overwrite. Reproduction must use a new
output path. The original checkpoint remains the baseline, and further work
must verify calibration with an independent source and improve ranking on
appropriate decision tasks before considering quantization or promotion.
