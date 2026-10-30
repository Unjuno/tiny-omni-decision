# EgoViz train-only temperature probe — 2026-10-30

The fixed [protocol](R2_EGOVIZ_TEMPERATURE_PROTOCOL_20261030.md) and code were
committed as `c2cf6d8a` before execution. The previous joint scorer, data and
selected checkpoint were hash-verified and left unchanged. No new readout or
encoder was trained. CPU inference reproduced the saved development probabilities
within `4.88e-8` and reproduced the selected training NLL.

One scalar was fitted by training NLL on 332 training examples. It was applied
once to all 307 development examples from 20 previously held-out sessions.
The development set also selected the original checkpoint, so this is an
exploratory development comparison. The group-held-out diagnostic additionally
fits its fold scalars using development labels and is not a deployable scalar.

| Condition | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Original selected scorer | 55.70% | 1.2190 | 0.6607 | 0.2235 | 0.3335 |
| Scalar fitted on training examples | 55.70% | 1.4737 | 0.6430 | 0.2189 | 0.7759 |
| Five-fold development-group diagnostic | 55.70% | 1.0699 | 0.5660 | 0.0348 | 0.5361 |

The train-fitted inverse temperature was **10.6163** (temperature 0.09419).
It raised training confidence from 39.21% to 91.83%, close to the in-sample
training Accuracy of 92.77%, but made development predictions overconfident.
Development NLL worsened by 0.2547; a paired 20-session percentile bootstrap
(2,000 draws, seed 17) gives baseline-minus-scaled NLL **-0.2547** with 95%
interval **[-0.4296, -0.0915]**. Brier improved by 0.0177, but its interval
[-0.0338, +0.0739] crosses zero. ECE changed little and its bin dependence
should not obscure the large confidence shift. Positive scaling left every
option ranking unchanged. Reject this train-fitted scalar for promotion.

The separate group-held-out development diagnostic used five temperatures,
with inverse temperatures 3.52–3.76. Its NLL and Brier improved compared with
the unscaled scorer. The corresponding NLL difference was +0.1491 with
conditional session-bootstrap interval [+0.0993, +0.1977]; Brier difference
was +0.0947 [+0.0702, +0.1203]. These intervals resample predictions with
already fitted fold scalars; they do not refit calibration in each bootstrap
replicate. The original scorer was selected using the same development set,
and these five scalars use development labels, so neither interval establishes
blind or deployed calibration. The result motivates evaluating a separately
fitted calibration method on genuinely independent session groups.

The train and development sample gap is consistent with in-sample overfitting
affecting the train-only calibration estimate, but it does not isolate its
cause. It does not establish a model-capacity ceiling, solve the 55.70%
ranking Accuracy, or establish four-modality decision quality. No calibration
or model is promoted and no sealed split was accessed.

## Reproduction and artifacts

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
& 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe' -m scripts.evaluate_egoviz_temperature --corpus 'C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02\full-aggregate' --selected 'C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02\readout-results-cpu-01' --output 'C:\CodexArtifacts\egoviz-temperature-20261030-attempt-01'
```

The original output already exists and the runner refuses to overwrite it.
Reproduction uses a new output directory and the input hashes in the result.
Python 3.11.9, NumPy 2.4.6 and PyTorch 2.6.0+cu124 ran on CPU; the process
completed in approximately eight seconds and used no paid compute.

- [Complete metrics and fixed fold assignments](../../manifests/pretrained_reuse/egoviz-temperature-results-20261030.json).
- `C:\CodexArtifacts\egoviz-temperature-20261030-attempt-01\predictions.jsonl`:
  307 rows, original and both scaled probability vectors;
  SHA-256 `9eed831c05d1fb1c818faec464ebce7e1f97dca315cc61dd076c1f0933018db0`.
- `scalar-reload-verification.json`: reapplying the saved scalars to original
  probabilities reproduced all saved vectors within `1.0e-15`; SHA-256
  `56af20576c5737b1cf9e0aeed415b05ed0401436add8fa3899539c54b956d2ca`.
- Frozen selected checkpoint SHA-256
  `b62c1cac0c38c459263dbc2c5ffd8eddc76b8654a2254a5f4d15fac8fe95a862`.
- Result SHA-256 `1a76721ab990d9f20071a77adaa6e378ba580f0519cb53afe1bcee009680e4d9`.

The bootstrap sample resamples sessions rather than individual overlapping
windows. Capture sessions are not known to be participant-independent.
No generated publisher action labels are used.
