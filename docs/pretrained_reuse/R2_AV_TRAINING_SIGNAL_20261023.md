# Conditional AV training-signal diagnosis — 2026-10-23

## Why this work was necessary

The October 23 source-screen/live-audit notes incorrectly described the
conditional AV readout as untrained, extrapolating the October 17 implementation
report. Actual October 22 artifacts prove that C0 and exchange/gate had completed
2,048 updates on CLEVRER synthetic AV, including a scalar-audio adaptation.
Both selected step 128 at 50% validation Accuracy / NLL 0.693147. Those old
results and artifacts remain unchanged. Their validation is exposed development
data, so this investigation uses TRAIN data exclusively for fitting and scoring.

## Checkpoint investigation

Pinned MiniLM option embeddings are distinct (cosine approximately 0.9403).
The two projected candidate vectors are much closer in the saved trained models:

| Model / stage | Projected candidate L2 distance | TRAIN probe Accuracy | TRAIN probe NLL |
|---|---:|---:|---:|
| Seed17 initialization | 4.538314 | 50% | 0.693203 |
| C0 best128 | 0.391041 | 50% | 0.693147 |
| C0 final2048 | 0.013425 | 50% | 0.693147 |
| Exchange/gate best128 | 0.300325 | 50% | 0.693147 |
| Exchange/gate final2048 | 0.014677 | 50% | 0.693147 |

These scores use 16 TRAIN scenes / 16 complete opposite-label pairs / 32 rows,
selected once by SHA-256(seed17:pair_id), one pair per scene. Each pair has exactly
equal question and video tensors but different audio and opposite targets.
Input and parameter gradients were measured; no optimizer update was applied to
the saved models. Final exchange/gate audio input gradient norm is 2.73e-9;
zeroing audio changes probabilities by at most 5.96e-8 on this cohort. This
documents a nearly input-insensitive state, but does not identify its cause.

## Tiny-cohort fitting control

Two fresh seed17 models used the identical frozen tensors and initial weights,
AdamW LR 0.001, weight decay 0.0001, norm clip 1, full batch32 and exactly256
updates. The only arm difference was freezing `baseline.query` at initialization.
There was no early stopping or checkpoint selection; only final weights were
saved and reloaded. Each arm consumed8,192 repeated example exposures from32
unique examples. The base encoders were never trained or loaded on the GPU.

| Updates | Original TRAIN Acc. | Original TRAIN NLL | Fixed-query TRAIN Acc. | Fixed-query TRAIN NLL |
|---|---:|---:|---:|---:|
| 0 | 50.00% | 0.693203 | 50.00% | 0.693203 |
| 16 | 50.00% | 0.693240 | 50.00% | 0.693214 |
| 64 | 50.00% | 0.693146 | 50.00% | 0.693145 |
| 128 | 65.625% | 0.654904 | 50.00% | 0.691651 |
| 256 | 100.00% | 0.00000439 | 93.75% | 0.086786 |

The original model's candidate distance recovers to21.9658. The fixed-query
distance stays4.5383, with exactly zero parameter change in that projection.
Both saved/reloaded final logits match exactly. Candidate-order controls differ
by at most3.58e-7 in probability. Trainable parameters are1,732,804 /1,658,500;
allocated model size is unchanged. Recorded fit+diagnostic time is15.07 /23.10s
on CPU, two threads. These times include diagnostics and are not hardware
benchmarks. No cloud or GPU compute was used.

**Interpretation:** the existing architecture and input path can fit this small
cohort without a query-freeze repair. Freezing the query is not justified as a
fix from this result. The earlier full-corpus collapse cannot be called an
unavoidable architecture limit, a broken gradient path, or insufficient
parameter capacity. Conversely, fitting32 examples proves no generalization or
AV complementarity; neither arm is a product candidate.

## Reproducibility and failed preflight

The first attempt stopped before any optimizer update because the option-vector
hash differed. Encoding on CPU with two threads instead of the original14 changes
rounding by up to2.384185791015625e-7. Restoring14 encoder threads reproduced the
original hash exactly. The hash check was retained; fitting uses two threads only
after encoding. The failure is preserved in
`C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-training-signal-20261023-01/preflight-failure.json`.

Successful run: `conditional-av-training-signal-20261023-02`, source commit
`543cfbf`, Python3.11.9 / PyTorch2.6.0+cu124, CPU only. Start/end:
2026-10-23T05:35:16.029125Z /05:35:54.897985Z.

```powershell
$env:PYTHONPATH='src;.'
python -m scripts.audit_av_training_signal `
  --features C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-coverage-v2-decision-features-attempt-01/decision-features.npz `
  --text-model-dir C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-minilm-pinned-cache-01 `
  --reference-run C:/CodexArtifacts/pretrained-reuse-r2/conditional-clevrer-av-scalar-audio-attempt-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-training-signal-20261023-02
```

Existing output directories are rejected. To reproduce, use a new directory.
The feature NPZ contains train and validation; arrays are decompressed when
accessed, then only selected TRAIN rows are used. It is not accurate to claim
that the container has no validation data. No validation row was fitted/scored;
no sealed audit file was opened. Old feature/config/checkpoint hashes were
verified before and after, with no changes.

| Artifact in successful output | SHA-256 |
|---|---|
| run-metadata.json | ff2cb47cfccd6e9675454632a3762771c3c3ae21c209f7e0dad544d5054a6c9c |
| results.json | 8aa57ad8e027cc28bc1829f174952beb845e8bf095aa124e89c1b6b07d008a05 |
| historical-train-probes.json | 9585c2fae74860a81767d531c70f5c34cc5adc8f868c864fcf85e122f7439d92 |
| diagnostic-final-original.pt | b54ff42e5c9bce0fd2cc885e7b9da590da054ead4c753db99bdf23439d8505ee |
| diagnostic-final-fixed_query.pt | b32ae91d80d2747f5b40bca26059bc3a474fa8971a8cbff5c02abbcbcc665313 |

## Prospective sampler follow-up

The small full-batch fit differs from the original large-corpus experiment in
several ways, so it does not establish that batching caused collapse. The next
matched experiment freezes the original3,696 TRAIN rows, initial seed17 weights,
scalar adapter, optimizer and2,048-update budget. It compares only independent
row shuffling with whole-pair shuffling, both at batch64. Every pair contains
opposite targets under the same question/video. Validation remains unused for
fitting/scoring; all training curves and final weights are retained. No result
will be treated as generalization quality or grounds for product promotion.
