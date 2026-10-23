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

## Matched sampler follow-up: completed

The small full-batch fit differs from the original large-corpus experiment in
several ways, so it does not establish that batching caused collapse. The next
matched experiment freezes the original3,696 TRAIN rows, initial seed17 weights,
scalar adapter, optimizer and2,048-update budget. It compares only independent
row shuffling with whole-pair shuffling, both at batch64. Every pair contains
opposite targets under the same question/video. Validation remains unused for
fitting/scoring; all training curves and final weights are retained. No result
is treated as generalization quality or grounds for product promotion.

Both arms completed 2,048 updates / 130,512 example exposures / 3,696 unique
rows from 773 scenes and 1,848 event pairs. Same 1,732,804 allocated/trainable
readout parameters; encoders remain frozen. Only final weights were saved, with
exact full-TRAIN logit reload. No best-checkpoint selection or validation scoring.

| Updates | Row shuffle TRAIN Acc. | Row shuffle NLL | Whole-pair TRAIN Acc. | Whole-pair NLL |
|---|---:|---:|---:|---:|
| 0 | 50.00% | 0.693203 | 50.00% | 0.693203 |
| 128 | 50.00% | 0.693147 | 73.27% | 0.653448 |
| 256 | 50.00% | 0.693147 | 74.76% | 0.476114 |
| 512 | 50.00% | 0.693228 | 75.35% | 0.465391 |
| 1024 | 50.00% | 0.693145 | 75.05% | 0.494656 |
| 1152 | 75.62% | 0.498248 | 75.62% | 0.467649 |
| 1536 | 75.38% | 0.462991 | 76.06% | 0.456969 |
| 1920 | 76.11% | 0.457813 | 75.35% | 0.539141 |
| 2048 | 75.38% | 0.461588 | 75.62% | 0.458694 |

Final Brier is 0.317146 / 0.316194; ECE-15 is 0.037916 / 0.024378. Whole-pair
batching learns sooner in this single CPU seed, but its endpoint difference is
only 9/3,696 rows (+0.24 percentage points). Do not select or promote an arm from
TRAIN results. NLL is not monotonic; neither convergence nor a quality ceiling
is established. Row shuffle's late departure from chance is a fitting transition,
not evidence of grokking on unseen data.

The reference comparison independently confirms exact feature, option-vector,
readout-source, and consumed-order hashes for row shuffle. The latter is
`74724968a38753e3358d77b34a88108706c9e547d48c716cabb375863d607b1f`,
the original GPU experiment's 130,512-index order. Yet this CPU run learns while
the old GPU run stayed near chance. Backend/numerical behavior is a material
confound for comparisons with the old run. The diagnostic runner and evaluation
schedule also differ (TRAIN measurements and gradient probes instead of
validation selection). No matched CPU/CUDA replay isolated the cause, so do not
attribute the old failure solely to batching or claim exact training replay.

## Fixed final-weight input controls

The final checkpoints were reloaded in a new CPU process. Original full-TRAIN
logits matched the saved arrays bit for bit. Controls leave weights/targets
unchanged: zero a stream, or replace video/question using a fixed half-cohort
cyclic shift. Every donor comes from a different scene. These mismatched-input
scores are dependency diagnostics, not valid new task labels or quality scores.

| Input condition | Row shuffle Acc. / NLL | Whole-pair Acc. / NLL |
|---|---:|---:|
| Original | 75.38% / 0.461588 | 75.62% / 0.458694 |
| Zero audio | 50.00% / 0.795533 | 50.00% / 1.048511 |
| Zero video | 75.38% / 0.464064 | 75.62% / 0.458694 |
| Different-scene video | 75.38% / 0.462748 | 75.62% / 0.458694 |
| Different-scene question | 75.38% / 0.461588 | 75.60% / 0.460586 |

Zeroing or swapping video changes **zero** final choices in either arm across
3,696 rows. Row-shuffle probabilities do respond to video (maximum change 0.155
for swapped video); whole-pair probabilities barely respond (maximum 1.37e-6).
Thus distinguish unchanged argmax from exact numerical invariance. Question
swaps change 0 choices in row shuffle and 13 in whole-pair. Audio removal returns
both arms to 50%. The learned decision remains dominated by audio timing.
These models do not demonstrate the requested AV evidence binding, even on
TRAIN. Faster fitting is insufficient grounds for adopting the sampler/model.

## Sampler execution, artifacts and next action

Source commit `39e4b0b`; full code hashes and exact arguments are in the protocol.
Local Intel i7-12700H, Python 3.11.9 / PyTorch 2.6.0+cu124, CPU FP32, two fitting
threads. Option encoding remains 14 threads. Start/end:
2026-10-23T05:39:11.173026Z / 05:48:56.833353Z. Arm times including measurements
and reload are 283.80 / 301.62 seconds. Controls take 16.76 seconds. These are
program wall times on a shared host, not isolated throughput benchmarks; peak
RAM was not instrumented. GPU/cloud compute and monetary cost are zero.

```powershell
$env:PYTHONPATH='src;.'
python -m scripts.audit_av_paired_sampler `
  --features C:/CodexArtifacts/pretrained-reuse-r2/clevrer-synthetic-av-coverage-v2-decision-features-attempt-01/decision-features.npz `
  --text-model-dir C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-minilm-pinned-cache-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-paired-sampler-20261023-01
python -m scripts.audit_av_sampler_controls `
  --run C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-paired-sampler-20261023-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/conditional-av-paired-sampler-controls-20261023-01
```

| File (under sampler output unless specified) | SHA-256 |
|---|---|
| run-metadata.json | 91f67ad293dedf74ae7ce9d96c65e2509be422969632c7bfdd204b7ffb888cd2 |
| results.json | 0566081b4e02d7d0a679859ee752ae70c5ed77b02965f67641742e6c3a8a0d87 |
| history-original.json | f21e3db6d32855d822bd8848cdc1c434a9a41723d270f9d996de03f36f79b68b |
| history-paired.json | ec4f13550ea1e26c596d06adaaf95571674330e9a4745e72dd57c111ec449ee4 |
| diagnostic-final-original.pt | d10b654d4c9f6715cbf0bba28514d9cfc73388c8a30541bb2f42fd0ab152f6b7 |
| diagnostic-final-paired.pt | 6f291111b4603b0154772d2c0b98384e154fc72f504c8fed64fe1052ee381fa7 |
| controls output/report.json | 50b9eeb49b0f9300b0a85acfdd58b63420756c82db20f55154daca30c500bf32 |

All final weights, predictions, complete curves, input identities and hashes
remain in new external artifact directories. Nothing overwrites prior v1/v2/QAT
models or corpus data. Orbitune was left running; no GPU reservation was needed.

Close this sampler comparison. Next, require a train-side task/control where
audio timing alone is insufficient and the requested collision/object binding
requires video, before another quality run. Preserve the temporal LSTM reference
and compare temporal binding paths only under such controls. A separate fixed
CPU/CUDA numerical replay could investigate the old collapse when the GPU is
cooperatively available; it must not tune to the exposed validation. No new
natural-AV quality, four-modality 95–98% claim, quantization, release qualification,
or product promotion follows from these diagnostics.

## Verification

The complete local CPU test suite passed: **2,551 passed, 11 skipped** (37.28s),
with CUDA hidden in the test process. Seven new tests cover TRAIN-only paired
selection, exact arm initialization/freeze policy, pair samplers, and immutable
input controls. Ruff passed for the whole repository. All 27 manifest/catalog
validation and audit commands from the CI workflow passed locally. GitHub Actions status is
reported against the published commit rather than inferred from local tests.
