# PCH temporal sensor readout probe — 2026-10-21

## Question and scope

Does a very small recurrent classifier over raw calibrated accelerometer windows
provide a useful sensor-only decision baseline beside frozen MOMENT features?
This tests the time-series path suggested for the larger Omni project; it is not
an audiovisual fusion or product-quality result.

The comparison reuses the previously frozen PCH Data 2017 participant split and
the exact 63 development-validation recordings used by the MOMENT probe. That
validation has already been used for checkpoint selection, so all intervals here
are development diagnostics, not independent generalization estimates. A separate
28-participant holdout was not opened. The source is the [PCH Data 2017 release](https://doi.org/10.21251/b9566edc-c68c-4eee-b67e-c0228d0d3833).

## Fixed comparison

- Training: 276 recordings from 92 participants; 13,100 nonoverlapping 2.5-second
  windows; 18 activity classes.
- Validation: 63 recordings from 21 disjoint participants; 3,041 windows.
- Fit and validation sample-order SHA-256:
  `32ba9659a2b6cec52812ddb4ac0bd60fab073f39ba92901d04c056102d12d3f9`.
- Heldout: 28 assigned participants, payloads not opened.
- Candidate: one-layer LSTM, hidden size 64, 18,834 trainable parameters. It reads
  each 128-sample, 3-axis window independently and averages window logits per
  recording. It does not model order between windows across a full recording.
- Train: seed 17, AdamW, learning rate 0.001, batch 64, max 100 epochs, patience
  10. Fit-only axis normalization; each recording has equal total loss weight.
  Best validation NLL checkpoint selected at epoch 31; early stopping at epoch 41.
- Reference: frozen AutonLab/MOMENT-1-small plus its linear head from attempt 01,
  35,346,642 total parameters. Its checkpoint, prediction and run-result hashes
  remain pinned to the earlier artifact.

The LSTM has higher point estimates on this split, but the participant-cluster
bootstrap intervals include zero for every paired quality difference:

| Model | Accuracy | Macro accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Window LSTM | 0.6508 | 0.6222 | 1.0117 | 0.4153 | 0.1188 | 0.7043 |
| Frozen MOMENT + linear | 0.6032 | 0.5972 | 1.0860 | 0.5627 | 0.1504 | 0.7243 |

There were 63 validation recordings. The LSTM got three more correct than MOMENT.
Paired bootstrap used 5,000 participant-cluster resamples, seed 17, and percentile
95% intervals. For **LSTM minus MOMENT**, the intervals were Accuracy
`[-0.1587, 0.2381]`, NLL `[-0.5369, 0.4320]`, and Brier `[-0.3535, 0.0622]`.
They all include zero; this small comparison does not establish superiority.
Per-class validation support was only 2–5 recordings per class.

## Learning curve and compute

| Epoch | Train recording accuracy | Validation accuracy | Validation NLL | Brier | ECE-15 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3152 | 0.2857 | 2.0939 | 0.8265 | 0.0753 |
| 10 | 0.6413 | 0.5238 | 1.5355 | 0.6165 | 0.1915 |
| 20 | 0.7971 | 0.5397 | 1.2719 | 0.5211 | 0.1695 |
| **31 selected** | **0.8768** | **0.6508** | **1.0117** | **0.4153** | **0.1188** |
| 41 final | 0.9384 | 0.6349 | 1.1027 | 0.4433 | 0.1469 |

After the selected epoch, training accuracy continued to rise while validation
accuracy, NLL, Brier and ECE degraded. The run shows overfitting; the final
checkpoint was kept separately from the selected checkpoint.

Execution used local RTX 3080 Laptop, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA
12.4, deterministic CUDA operations and FP32 weights. Training and validation
completed in 33.31 seconds; full measured run was 41.25 seconds. Peak CUDA
allocated/reserved memory was 137,250,816 / 165,675,008 bytes. The selected
checkpoint is 78,916 bytes. Cloud cost was $0. No model weights or media were
added to Git. The training run was launched through the HotUGPU guard with a
5,120 MiB reservation and 8,000 MiB minimum child-free gate; the effective run
receipt measured 11,095 MiB free before the child started. The command shape was:

```text
python -m scripts.run_with_hotgpu_guard --max-wait-seconds 3600 --reserve-mib 5120 --block-if-free-mib 14000 --min-child-free-mib 8000 -- <Python 3.11 executable> -m scripts.pch_lstm_sensor_probe run --archive <PCH_Data_2017.zip> --reference-predictions <attempt-01/validation-predictions.jsonl> --reference-result <attempt-01/result.json> --output-dir C:/CodexArtifacts/pretrained-reuse-r2/pch-lstm-sensor-attempt-04
```

HotUGPU initially rejected attempts because utilization telemetry briefly rose
after the reservation allocation even when there were no external compute PIDs.
A one-time child diagnostic measured 11,095 MiB free, the reserving parent as the
only GPU PID, and a stale 34% utilization sample. The guard was changed to retain
the VRAM reservation until utilization is below 5% for two consecutive samples;
it releases the reservation if an external PID or memory-gate violation appears.
The corrected guard passed its CPU tests. The successful run used a 5 GiB
reservation, leaving 11,095 MiB for the child. Orbitune was never stopped or
signalled.

## Artifact integrity and correction

The experiment code is commit `9a6ea9b973044ae2e724cd5bd4ed971801ff824b` on
`codex/r2-pch-temporal-readout-v1`; the committed training-script SHA-256 is
`90bd4af04cd86d7305aa13546cafef58778ae710120aa18aacdd32c587e57916`. The
effective config's canonical SHA-256 is
`39b9e727009f09335f3d378a8b94d254247e62f390f10a17d5a507cf3a02005e` (the
serialized file SHA-256 is
`eb4d642d7c22fcd6e44df803e0bb64ad2c069354010e481a159989b01fc39814`); the
preflight SHA-256 is
`963ad2c1517e04a40015b0b32c7a36dae9ab48fc5de3797e325667939af5136f`. An
external manifest pins every run file and source snapshot:
`artifact-hashes.json` SHA-256
`7f11ed0e15e81bd3db35119338263a7518f03f2b588085f24341961aee896953`.

The first result receipt reused a MOMENT-versus-static bootstrap helper whose
embedded direction string named the wrong models. No training or inference was
repeated. The intervals above were recomputed from saved predictions by
[`report_pch_lstm_bootstrap.py`](../../scripts/report_pch_lstm_bootstrap.py),
and saved separately in `bootstrap-correction.json`; that correction records the
original direction string and pins the source result and predictions by hash.
Use the correction artifact for this comparison's interval labels.

All experiment files are under
`C:/CodexArtifacts/pretrained-reuse-r2/pch-lstm-sensor-attempt-04/`, including
effective config, preflight, run metadata, epoch history, best/final checkpoints,
saved validation probabilities, the corrected bootstrap and artifact hashes.
The original MOMENT attempt-01 config, feature cache, checkpoint and result were
read-only and unchanged. Attempt-02 was a preliminary preflight without its
baseline-artifact pins; attempt-03 stopped at the HotUGPU child preflight. Neither
attempt performed training or loaded heldout payloads.

## Interpretation

This is evidence that a sub-20K-parameter recurrent window classifier can produce
competitive point estimates on one narrow wearable-activity split. It does not
show a statistically resolved improvement over frozen pretrained features, does
not establish long-range temporal reasoning, does not reach the 95% target, and
does not validate audio/video/image/text or sensor fusion. The current result
supports keeping a small temporal readout as an architecture candidate. Any next
comparison should use a fresh participant-grouped development split or a new
rights-cleared sensor task and must report validation selection and uncertainty.

