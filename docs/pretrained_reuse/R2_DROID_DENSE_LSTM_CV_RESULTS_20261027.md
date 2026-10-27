# Small LSTM on fixed DROID train-window folds

The [fixed follow-up](R2_DROID_DENSE_LSTM_CV_PROTOCOL_20261027.md) trained a
one-layer LSTM with eight hidden units and a seven-choice decision head on the
same 80,998 windows, five recording-group folds, change-weighted loss, seeds,
batch size and 50 epochs as the preceding linear comparison. Both receive the
same previous command and eight past sensor states. The LSTM/head has **880**
trainable parameters, versus **840** for the prior linear head. The LSTM uses
seeded default initialization and coordinate-wise temporal normalization;
the linear head used zero initialization and standardized last-state/delta
features. Thus this is a near-size exploratory comparison, not an isolated
effect of recurrence alone.

| Model / probabilities | Accuracy | NLL | Macro NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Linear, weighted | 0.84734 | 0.82747 | 0.88677 | 0.38132 | 0.30934 |
| LSTM, weighted | **0.87574** | 0.87748 | 0.94095 | 0.40256 | 0.38018 |
| Linear, fixed probability correction | 0.90696 | **0.44286** | **0.48758** | 0.17430 | 0.01779 |
| LSTM, fixed probability correction | 0.90812 | 0.45328 | 0.49795 | 0.17445 | 0.02298 |

| Model / probabilities | Change detection recall | Change precision | Stable false-change rate | Exact changed command |
|---|---:|---:|---:|---:|
| Linear, weighted | 14.53% | 16.76% | 7.30% | 444 / 7,442 |
| LSTM, weighted | 6.07% | 13.99% | 3.78% | 156 / 7,442 |
| Linear, corrected | 0.44% | 24.44% | 0.139% | 8 / 7,442 |
| LSTM, corrected | 0% | undefined | 0% | 0 / 7,442 |

Detection means predicting any command different from the previous one, not
necessarily the correct new command. The LSTM reduced false alarms and
increased overall Accuracy, but detected fewer changes and had worse NLL,
Brier and ECE under both reported probability conventions. Its corrected
argmax always retained the previous command. These are not calibrated product
operating points; correction is algebraic and was not fitted.

For weighted raw LSTM minus linear, a paired bootstrap of 275 recording groups
(2,000 resamples, seed 17, percentile 95%) gave Accuracy +0.02840
[+0.01342, +0.04450] and NLL +0.05001 [+0.03645, +0.06295]; higher NLL is
worse. This describes the already inspected train-CV cohort, not independent
operator/site generalization. Windows within recordings are correlated, and
axis-label changes can include small command fluctuations.

Mean weighted training loss across folds at epochs 1 / 10 / 20 / 30 / 40 / 50
was 1.915 / 1.517 / 1.451 / 1.429 / 1.416 / 1.409. It was still falling; no
held learning curve was measured, so neither a plateau nor lack of overfitting
is established. This one small configuration and fixed budget do not show
that LSTM, xLSTM, longer training or a different task cannot help. No video,
audio, validation, sealed audit or product checkpoint was used.

Synthetic CPU smoke measured 880 parameters and approximately 0.00487 seconds
per batch after warmup. Actual five-fold training and aggregation took 45.49
seconds on CPU with PyTorch 2.6.0+cu124 and four threads; no GPU was used.
Independent replay checked all five heads, train-only normalizers and fold
weights: raw probability maximum absolute difference was 0 and corrected
maximum difference `2.99e-8`. The final product quality goal remains open.

Private output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-dense-lstm-cv-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`9bd1afbc137aa0d66903bb695bffcf6c37eecb0aac5a679c403953b31c859c9c`;
`oof-probabilities.npz`:
`faf346ae125a87268f23d0280dc2dcf9b62b00537c96ad5caed9353f0964acec`.
The summary includes five model hashes, training curves, all metrics, fold
counts, bootstrap intervals and dataset/code/protocol provenance.
