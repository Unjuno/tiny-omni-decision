# R2 synthetic AV temporal/readout comparison — 2026-10-25

## Result and scope

The fixed-seed experiment completed on the full generated corpus. A two-layer
temporal LSTM learned the sequence-matching rule after an extended chance-level
period, then reached 100% on the 1,024-row generated validation cohort. A
parameter-matched query-conditioned bidirectional cross-attention readout with
a learned modality gate remained at chance after the same 16,384 optimizer
updates.

This is a **synthetic architecture diagnostic only**. It does not measure
natural video understanding, speech understanding, general multimodal decision
quality, or the 95–98% product objective. The generated train and validation
splits reuse the same four visual-symbol and four tone prototypes by design;
they hold out complete sequence pairs, not symbol vocabularies or acoustic
domains. The validation cohort was used for checkpoint selection and is not a
blind test.

## Frozen protocol

- Experiment: `synthetic-av-temporal-vs-conditional-v1`; seed 17.
- Task: determine whether two eight-step audio/video symbol sequences have the
  same order. Each group contains four pairings (`AA`, `BB`, `AB`, `BA`) and
  both query polarities, for eight rows. Same/different-order sequences have
  matching symbol multisets.
- Splits: 512 train groups / 4,096 rows and 128 validation groups / 1,024 rows.
  There are 1,024 unique train and 256 unique validation video sequences, and
  1,280 distinct whole-video plus 1,280 distinct whole-audio sequence assets.
  No group, sequence identity, or complete media asset crosses splits. The four
  single-symbol tone prototypes and four glyphs are shared intentionally.
- Features: pinned, frozen VideoPrism base (`[N,8,768]`) and EfficientAT
  DymN04-AS (`[4,384]`, expanded into eight aligned event positions). No encoder
  weights were updated.
- Both heads received the same seed, epoch permutations, rows, AdamW settings
  (LR 0.001, weight decay 0.0001), batch size 32, and 128 epochs. Validation
  ran every four epochs; selection used minimum validation NLL with the earliest
  exact tie retained. The terminal checkpoint is separate from the best.
- The preflight SHA-256 for the complete 128-epoch permutation stream is
  `7988a28d484c35ca071cc07a57e9f8c509860e28c175aa7e49d097ee0afe60e9`.
  Each head saw all 4,096 unique train rows once per epoch, 16,384 updates, and
  524,288 example presentations. There was no within-epoch replacement.

## Frozen encoders and runtime

| Component | Revision / checkpoint | Parameters | Feature device and runtime |
| --- | --- | ---: | --- |
| VideoPrism base | source `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | 114,365,184 | Windows JAX CPU (`TFRT_CPU_0`), JAX 0.6.2 / TensorFlow 2.18.1; full extraction 2,281.21 s |
| EfficientAT DymN04-AS | source `a425fdce92572e602a1d5634799bd9f1f2efa806`; checkpoint SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | 1,961,239 | CPU feature extraction 0.278 s; frozen state hash before and after matched |

The training host was an NVIDIA GeForce RTX 3080 Laptop GPU (16,384 MiB),
Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4. All CUDA training used
`scripts/run_with_hotgpu_guard.py`; it waited for a cooperative yield, reserved
2,418 MiB, then started the child. The first invocation failed before GPU
reservation because the guard requires an absolute executable path; retrying
with the resolved Python path succeeded. No competing process was stopped.

## Readout learning curves

Validation metric selection was fixed to NLL. Train metrics below are online
epoch measurements while weights change and are **not** a fixed-checkpoint
generalization-gap estimate. The complete 32-point per-head history is in the
external run artifact.

| Readout | Step | Validation Accuracy | Validation NLL | Validation Brier | Validation ECE | Online train CE / Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 512 | 0.5000 | 0.693156 | 0.500009 | 0.002097 | 0.693597 / 0.4956 |
| LSTM | 2,048 | 0.5000 | 0.693158 | 0.500010 | 0.002302 | 0.693307 / 0.4888 |
| LSTM | 3,584 | 0.5000 | 0.692667 | 0.499520 | 0.002563 | 0.694079 / 0.4995 |
| LSTM | 4,096 | 0.7275 | 0.455375 | 0.313188 | 0.136379 | 0.549542 / 0.7029 |
| LSTM | 4,608 | 0.9795 | 0.051775 | 0.028311 | 0.015887 | 0.127464 / 0.9619 |
| LSTM | 5,120 | 0.9990 | 0.005456 | 0.001537 | 0.003748 | 0.016649 / 0.9961 |
| LSTM | 6,144 | 1.0000 | 0.002402 | 0.000755 | 0.002144 | 0.001893 / 0.9998 |
| LSTM | 8,192 | 0.9990 | 0.001651 | 0.001109 | 0.001123 | 0.000044 / 1.0000 |
| LSTM, selected best | 11,264 | 1.0000 | 0.000585 | 0.000277 | 0.000494 | 0.000003 / 1.0000 |
| LSTM, terminal | 16,384 | 1.0000 | 0.000935 | 0.000451 | 0.000788 | 0.000039 / 1.0000 |
| Conditional AV | 512 | 0.5000 | 0.693249 | 0.500102 | 0.007135 | 0.693895 / 0.5015 |
| Conditional AV | 2,048 | 0.5000 | 0.693173 | 0.500026 | 0.003571 | 0.693419 / 0.4863 |
| Conditional AV, selected best | 3,584 | 0.5029 | 0.693147 | 0.500000 | 0.002908 | 0.693280 / 0.4971 |
| Conditional AV | 8,192 | 0.5000 | 0.693148 | 0.500000 | 0.000476 | 0.693199 / 0.4863 |
| Conditional AV | 12,288 | 0.5000 | 0.693147 | 0.500000 | 0.000382 | 0.693205 / 0.4951 |
| Conditional AV, terminal | 16,384 | 0.5000 | 0.693148 | 0.500001 | 0.000506 | 0.693207 / 0.5000 |

The LSTM curve contains a clear delayed transition: validation remained at
chance through step 3,584, improved sharply at step 4,096, and reached 97.95%
at step 4,608. This supports a training-time/grokking-like explanation for
that head on this synthetic rule. It does not establish grokking or a learning
curve for natural AV tasks. The conditional head stayed near chance on both
training and validation metrics throughout the full budget; this is not
evidence of overfitting.

## Selected-checkpoint validation and input controls

Metrics below are recomputed from the saved per-example probabilities and the
saved `logits.argmax` predictions. Each cohort has 1,024 examples and 128
groups. `pair exact` means both complementary-query rows for a pair are
correct.

| Readout / condition | Accuracy | NLL | Brier | ECE | Mean confidence | Pair exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM, full aligned AV | 1.0000 | 0.000585 | 0.000278 | 0.000493 | 0.999507 | 1.0000 |
| LSTM, audio only | 0.5000 | 8.504574 | 1.000000 | 0.500000 | 1.000000 | 0.5000 |
| LSTM, video only | 0.5000 | 7.503171 | 0.990297 | 0.493650 | 0.993650 | 0.4609 |
| LSTM, both missing | 0.5000 | 4.498039 | 0.963085 | 0.480806 | 0.980806 | 0.5000 |
| LSTM, audio paired to another row in same group/query | 0.5000 | 7.728733 | 1.000100 | 0.500398 | 0.999507 | 0.5000 |
| LSTM, reverse audio time only | 0.5088 | 8.066625 | 0.979236 | 0.490054 | 0.998501 | 0.5078 |
| LSTM, reverse both streams in time | 0.8018 | 1.919118 | 0.381809 | 0.190371 | 0.988198 | 0.7090 |
| Conditional AV, full aligned AV | 0.5000 | 0.693147 | 0.500000 | 0.000022 | 0.500022 | 0.2813 |
| Conditional AV, audio only | 0.5000 | 0.693147 | 0.500000 | 0.000024 | 0.500024 | 0.5000 |
| Conditional AV, video only | 0.5000 | 0.693147 | 0.500000 | 0.000021 | 0.500021 | 0.5000 |
| Conditional AV, both missing | 0.5000 | 0.693147 | 0.500000 | 0.000023 | 0.500023 | 0.5000 |
| Conditional AV, shuffled audio pairing | 0.5020 | 0.693147 | 0.500000 | 0.001931 | 0.500022 | 0.2832 |
| Conditional AV, reverse audio time only | 0.5000 | 0.693147 | 0.500000 | 0.000022 | 0.500022 | 0.2813 |
| Conditional AV, reverse both streams in time | 0.5020 | 0.693147 | 0.500000 | 0.001931 | 0.500022 | 0.2813 |

The LSTM needs the joint, correctly aligned streams on this rule: each
single-modality and shuffled-pair control is at 50% despite extreme confidence,
while the intact pair is perfect. Reversing the audio stream alone returns it
to chance; reversing both still scores 80.18%, so the model is not perfectly
invariant to a shared time reversal. The conditional readout's predictions
remain near uniform regardless of input intervention, consistent with its
chance-level learning curve.

On the selected best checkpoints, the paired bootstrap over 128 complete
sequence-pair groups (10,000 replicates, seed 17, percentile 95% interval)
estimates conditional-minus-LSTM group-macro Accuracy as -0.5000, interval
[-0.5029, -0.4971]. This is a descriptive interval on a validation set used
for checkpoint selection, not an independent test interval.

## Metric reconciliation

The one-time evaluator originally computed Accuracy from
`softmax(logits).argmax`, while it serialized predictions from `logits.argmax`.
For a few near-zero conditional logits, float32 softmax rounded the two
probabilities to or across a tie, changing Accuracy/ECE relative to the saved
class prediction. The model was **not** re-evaluated. The separate CPU replay
uses the saved class prediction for Accuracy/ECE/pair/group metrics and saved
probabilities for NLL/Brier/confidence. It verified 28 cohorts; the only
reconciled fields were Accuracy/ECE on three conditional-best controls. This
also led to a code fix so future metric computation uses `logits.argmax`
consistently. The original evaluation JSON and predictions remain untouched.

Replay implementation follow-up: attempt-05 was incorrectly marked failed
because the replay verifier assumed every expected field had been computed
from probability argmax. Inspection of the original evaluator showed that
pair/group fields came from the saved `logits.argmax` predictions while
Accuracy/ECE came from probability argmax. Attempt-05 is retained as a failed
verifier diagnostic and is not evidence of a model/evaluation failure. The
verifier now checks each field against its actual provenance and rejects
partial saved-prediction columns. Fresh CPU replay attempt-06 passed all 28
cohorts, with no unexplained mismatches and the same three documented
Accuracy/ECE corrections. Its receipt is external at
`C:\CodexArtifacts\pretrained-reuse-r2\synthetic-av-readout-replay-v1-attempt-06\replay.json`
(SHA-256 `941dfedbc3673ad351db2f9fdf5eb42f36a398d111c22c1f981ded8e798bdeac`).

## Implementation corrections before the measured run

- The first training smoke found that the parameter-match tolerance had been
  nested under the model map and was parsed as a third architecture. The config
  was corrected, that output was marked `FAILED`, and a later CPU smoke passed
  before the full run.
- An initial preflight summary read encoder runtime fields from the merge
  receipt and left them null. It was superseded by `preflight-v2`, which reads
  the separate VideoPrism and EfficientAT feature receipts. The full run used
  the same corpus and feature-cache hashes listed here.
- The first HotGPU invocation used the unresolved command name `python` and
  failed before reserving the GPU. The absolute-path retry is the run described
  in this report.

## CI follow-up

The first pushed CI attempt failed during CPU-only test collection because the
new synthetic integration tests imported optional PyTorch and PyAV dependencies
at module load time. The failure was reproduced in an isolated environment
containing only the repository `.[dev]` dependencies. Tests that require those
optional integrations now skip cleanly when the dependency is absent, while
CPU-only data, metric, and replay tests remain enabled. In that exact minimal
environment, the suite passed with 1,856 passed and 162 skipped. The full local
environment passed with 2,491 passed and 11 skipped; Ruff and manifest/catalog
validation also passed. GitHub Actions for fix commit `92b996b` completed
successfully. The earlier failed attempts are retained in branch history.
## Parameters, compute, and artifacts

| Readout | Trainable readout parameters | Full frozen encoders + readout parameters | Training wall time | Peak CUDA allocated / reserved |
| --- | ---: | ---: | ---: | ---: |
| Temporal LSTM | 555,458 | 116,881,881 | 64.91 s | 502,932,992 / 522,190,848 bytes |
| Conditional AV | 513,412 | 116,839,835 | 242.63 s | 349,546,496 / 532,676,608 bytes |

The comparison heads differ by 7.58% in trainable parameters. Total training
time was about 307.54 seconds; full VideoPrism extraction took 2,281.21 seconds
on CPU. No cloud compute or paid resources were used. The heads are research
readouts over offline features, not a shipped or quantized model. No tensor or
media file was added to Git.

External immutable attempt paths:

- Rendered synthetic data: `C:\CodexArtifacts\pretrained-reuse-r2\synthetic-av-readout-data-v1`
- Video / audio feature caches: `...\synthetic-av-readout-video-v1` and `...\synthetic-av-readout-audio-v1`
- Aligned full cache: `...\synthetic-av-readout-aligned-v1`
- Preflight: `...\synthetic-av-readout-preflight-v2\preflight.json`
- Training run, history, best and terminal weights: `...\synthetic-av-readout-training-v1-attempt-01`
- One-time validation and per-row predictions: `...\synthetic-av-readout-eval-v1-attempt-01`
- Independent replay and metric corrections: `...\synthetic-av-readout-replay-v1-attempt-04\replay.json`

Important hashes:

- Source commit used for training: `2c0c4620fb66ed0fc281c0a628135b20976523da`
- Readout config: `b1935fdb7b542df0e62d4a07b44303b870279e7b9ac7559dc57afda7a5e7bfa5`
- Frozen feature config: `95e954357b8aa1ccc16803577f00b0327e6babdc9375a14b556ca86879c1b909`
- Render manifest: `19dabb8935628d90ebd1b843148ed174ee49a931c4ca61b687edc415b67cde5`
- Aligned feature cache: `0685402a5a5d4603f8c9f9dabaae7da94f5ef6ab0cc38fa76323385f1e82db16`
- Training run metadata: `5e128e7c0147d896a057815a3cf657c2b2d761b5d8c3607ba4f1d0ac0de222b3`
- Validation predictions: `f83a420e16ccae790d4fcdb150904bb8e801092f3c575dfabc7834eee7ffa212`
- Independent replay attempt-04: `5e700da53f7b4a8976670a8a2a92f529608f503d3f97e22cd409d118b6162ccd`
- Independent replay attempt-06: `941dfedbc3673ad351db2f9fdf5eb42f36a398d111c22c1f981ded8e798bdeac`

## Interpretation and next step

The strongest finding is architectural and task-bounded: a modest temporal
LSTM learned a relational sequence rule from the frozen AV features, but the
current bidirectional cross-attention-plus-gate readout did not learn it under
the same budget. The result supports continuing sequence-aware decision
experiments, while rejecting this particular conditional head as a winner.
It does not prove that the head family is incapable or that the LSTM transfers
to natural AV data.

Before any product-facing AV training, the next useful diagnostic is to inspect
the conditional head's optimization path (projection/attention/gate gradient
scales and feature normalization) and compare a simpler conditioned temporal
fusion head on the same generated task. Keep this run frozen as the baseline;
do not change its seed, training history, checkpoints, corpus, or selector. Any
new variant should have a new config and output path. Natural-media transfer
still requires a separate rights-cleared, group-disjoint validation cohort.
