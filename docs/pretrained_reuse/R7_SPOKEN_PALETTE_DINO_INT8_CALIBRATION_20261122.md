# R7 INT8 calibration-only temperature fit — 2026-11-22

## Protocol

After the fixed dynamic-INT8 DINOv2 compatibility run, one scalar temperature
was selected on the already frozen Calibration split only. The selector exactly
matches the pilot's prior procedure: minimum macro-color NLL over 81
log-spaced temperatures in `[0.25, 4.0]`, with the earliest temperature on
ties. The quantized DINO state, float32 Decision scorer, corpus, labels,
normalizer and modalities were unchanged. Evaluation JSONL was not opened and
no sealed audit was accessed.

The Calibration split contains 14,280 decision rows over four source-TRAIN
speaker groups and 24 unique generated screens. Input hashes are recorded in
the external run receipt at
`C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-int8-calibration-attempt-01\result.json`:
corpus manifest
`b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a`,
calibration rows
`73965ed05ec47be9964acea70f3f8635cd2618c0adff0dd5f7e3540736377f0a`, INT8
state `cd8083402b68ed41f9a3367258698ba7cc70c455b58f31828e29346e281381d8`,
and float32 scorer
`48b4a12401f9da88472b290536ca5e3fe3b708319af9d221caa5ce4b75e9c1b5`.
The run's exact runner bytes are pinned by the adjacent `source-receipt.json`
with SHA-256
`f6afcfa3398335fdd6ce0b00157f5933724759876f5e1d236fbe09e264849976`;
they were executed as an uncommitted script based on commit
`1fa88f47fb2b14c30bc8a74f896d5417c40cfc8b` and are now checked into this
branch unchanged.

## Calibration-set diagnostics

| Calibration-only metric | Existing temperature 0.90125 | Selected temperature 0.40613 |
|---|---:|---:|
| Accuracy | 97.7171% | 97.7171% |
| Macro-color NLL | 0.13324 | 0.07538 |
| Brier | 0.05080 | 0.03416 |
| ECE-15 | 0.07316 | 0.00665 |
| Mean confidence | 0.90447 | 0.97474 |

These are in-sample calibration diagnostics: temperature was selected on these
same rows. They are not an independent quality estimate. The unchanged
argmax/Accuracy is expected because positive scalar temperature preserves
candidate ordering. A held-out post-calibration evaluation was intentionally
not performed; the pilot's original development evaluation has already been
used and must not be re-opened for this purpose.

The CPU-only run used PyTorch 2.6.0+cu124, four threads, and completed in 4.86
seconds. It loaded and verified the existing dynamic-INT8 DINO state, recomputed
features only for Calibration screens, and saved per-row Calibration
predictions and hashes. No training, GPU, cloud, or model-weight mutation took
place.

## Interpretation

This confirms that the observed INT8 confidence shift can be addressed on the
reserved Calibration partition for this fixed synthetic task. It does **not**
show that the new temperature generalizes, restores probabilities on untouched
examples, or improves a natural multimodal task. It also does not quantify a
complete model package: cached Whisper features and a float32 scorer were used.
The next honest calibration-quality claim requires a new untouched evaluation
cohort from a separately admitted source or a preregistered sealed audit;
neither may be substituted by reusing these examples.
