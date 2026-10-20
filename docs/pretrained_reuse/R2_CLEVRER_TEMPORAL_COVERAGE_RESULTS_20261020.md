# CLEVRER temporal-only fresh-coverage result (2026-10-20)

## Outcome

**Execution: PASS.** Both matched readouts completed 2,016 optimizer updates on
the same frozen, scene-disjoint corpus. Best-checkpoint reloads reproduced the
validation logits bit-for-bit; best and final checkpoint files remain separate.
No sealed audit, other modality, model encoder, or existing Teacher artifact was
loaded or modified.

This is a narrow text-free video Decision experiment. Its validation accuracy
is about 37–40%, far below the project's 95% task-quality goal. It does not
support a product-quality claim, a general video capability claim, or a
capacity ceiling. The validation set selected checkpoints, so it is a
development result, not a blind estimate.

## Data and execution

- Source: MIT-IBM/CLEVRER train split, revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0.
- Unique training coverage: 3,526 question rows / 324 video scenes and assets.
- Validation: 321 rows / 81 video scenes and assets.
- Question family: temporal descriptive only. The model sees VideoPrism
  framewise features, the typed CLEVRER program, taxonomy and supplied options;
  natural-language questions are excluded from model inputs.
- Train/validation overlap: zero scenes, zero media hashes, zero normalized
  question/options. Validation is not sealed and was used for selection.
- Seed 17; batch size 64; 36 shuffled full-data epochs; 56 updates per epoch;
  constant AdamW LR 0.001, weight decay 0.0001, gradient clip 1.0; no early
  stopping. Each arm consumed 126,936 example exposures, 3,526 unique examples
  and 324 unique video assets. Selection was minimum macro taxonomy validation
  NLL with the earliest tie.
- Source commit: `722b5a7c5fce02784921163ff1592b74938e11a9`.
- Config SHA-256:
  `164bec7c4129ba024e07d1f00f274e5d7b02c4e2189ab8d4793cd846bc1523e3`.
- Frozen train/validation ID-order hashes and source receipts are recorded in
  [the experiment protocol](R2_CLEVRER_TEMPORAL_COVERAGE_PROTOCOL_20261020.md)
  and the external run's `input-preflight.json`.

The feature encoder was VideoPrism-B, pinned revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
Eight `[768]` frame-position features were generated for each of 405 videos
with the existing JAX 0.6.2 CPU environment in 604.89 seconds; peak working set
was 4,174,123,008 bytes. The feature-cache SHA-256 is
`e3008bfa3c672fb52f3c21b4dcdd8f9f7b5c383f9f774a8afec042429c1cbbcb`.

The RTX 3080 Laptop GPU ran only the small readout fits through the
HotUGPU cooperative reservation. The wrapper waited while Orbitune owned the
GPU, then reserved 3,200 MiB after its yield and launched the child with
13,015 MiB free. Orbitune was not stopped or signaled. Total readout wall time
was 27.87 seconds; mean-MLP / BiLSTM training time was 10.00 / 11.11 seconds,
validation/evaluation time 2.03 / 1.93 seconds, and peak allocated CUDA memory
318,105,088 / 481,556,992 bytes. Cloud cost was $0.

## Model comparison

Metrics in the first table are macro averages across the five taxonomy groups.
The selector is macro taxonomy NLL. Overall metrics are separately shown to
make the class-count weighting visible.

| Readout | Trainable params | Best update | Macro accuracy | Macro NLL | Macro Brier | Macro ECE-15 | Overall accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean MLP | 226,865 | 1,344 | 0.3709 | 1.1706 | 0.6456 | 0.1579 | 0.3614 |
| BiLSTM | 227,377 | 1,232 | 0.3963 | 1.1667 | 0.6431 | 0.1525 | 0.3832 |

Paired scene-cluster bootstrap for selected BiLSTM minus selected mean-MLP
(5,000 resamples, seed 17; percentile 95% intervals; 81 scenes): Accuracy
+0.0218 [−0.0213, +0.0659], NLL −0.00017 [−0.0310, +0.0313], Brier +0.00020
[−0.0172, +0.0177]. All intervals include zero; this run does not establish a
recurrent-readout advantage.

Selected validation metrics by taxonomy:

| Taxonomy (n) | Mean-MLP Accuracy / NLL / Brier / ECE | BiLSTM Accuracy / NLL / Brier / ECE |
|---|---:|---:|
| Count (86) | 0.3140 / 1.3248 / 0.7095 / 0.1357 | 0.3140 / 1.3617 / 0.7389 / 0.1826 |
| Exist (80) | 0.4125 / 0.7526 / 0.5577 / 0.1938 | 0.4375 / 0.7360 / 0.5417 / 0.1575 |
| Query color (58) | 0.1379 / 2.1121 / 0.8853 / 0.1335 | 0.1552 / 2.1029 / 0.8817 / 0.1251 |
| Query material (46) | 0.5000 / 0.6953 / 0.4976 / 0.2484 | 0.5652 / 0.6541 / 0.4632 / 0.1315 |
| Query shape (51) | 0.4902 / 0.9683 / 0.5779 / 0.0781 | 0.5098 / 0.9787 / 0.5898 / 0.1657 |

## Learning curve and overfit

Each point is a fixed full-training-set evaluation and the same 321-example
validation set. Accuracy and NLL below are macro taxonomy metrics. Training
values are not rolling-batch loss estimates.

| Readout | Update | Train accuracy / NLL | Validation accuracy / NLL / Brier / ECE |
|---|---:|---:|---:|
| Mean MLP | 504 | 0.3859 / 1.1901 | 0.3426 / 1.1939 / 0.6578 / 0.0892 |
| Mean MLP | 1,008 | 0.4646 / 1.1143 | 0.3520 / 1.1916 / 0.6643 / 0.1391 |
| Mean MLP | 1,344 (selected) | 0.5101 / 1.0525 | 0.3709 / 1.1706 / 0.6456 / 0.1579 |
| Mean MLP | 1,512 | 0.5293 / 1.0291 | 0.4168 / 1.1803 / 0.6437 / 0.1496 |
| Mean MLP | 2,016 (final) | 0.6427 / 0.8581 | 0.4083 / 1.2452 / 0.6755 / 0.1801 |
| BiLSTM | 504 | 0.3799 / 1.1910 | 0.3364 / 1.1932 / 0.6574 / 0.0865 |
| BiLSTM | 1,008 | 0.4730 / 1.0992 | 0.3652 / 1.1878 / 0.6612 / 0.1297 |
| BiLSTM | 1,232 (selected) | 0.5314 / 1.0328 | 0.3963 / 1.1667 / 0.6431 / 0.1525 |
| BiLSTM | 1,512 | 0.5915 / 0.9193 | 0.4291 / 1.1759 / 0.6423 / 0.1482 |
| BiLSTM | 2,016 (final) | 0.7695 / 0.6136 | 0.3702 / 1.3442 / 0.7346 / 0.2658 |

Both training curves continue to fit the training examples while final
validation NLL, Brier and calibration deteriorate. This is clear overfitting
within this 36-epoch budget. The endpoint accuracy alone is not a valid reason
to replace the NLL-selected checkpoint.

## Video and temporal controls

At each validation-selected checkpoint, zeroing or scene-shuffling the video
features reduced accuracy and worsened NLL. For mean-MLP, reversing frame order
is exactly invariant by construction: 0.3709 macro accuracy / 1.1706 NLL.
For BiLSTM, the aligned input scored 0.3963 / 1.1667 and reversed frames scored
0.4039 / 1.1694. The small, mixed change does not demonstrate a useful ordered
temporal signal. Video features affect the choice scores, but the task result
does not show that recurrent temporal order is the missing capability.

## Interpretation and next step

This experiment expanded training coverage from the prior text-free CLEVRER
probe's 512 rows / 256 scenes to 3,526 temporal-descriptive rows / 324 scenes.
However, it also uses a new temporal-only cohort and new validation scenes, so
the comparison cannot isolate the causal effect of added data versus the older
mixed-task run. It does show that the present small text-free readout does not
solve this fresh temporal-description task to useful accuracy and overfits as
training continues. The mean-MLP and BiLSTM selected scores are statistically
indistinguishable at this sample size.

Do not respond by extending this same validation-selected run, changing
architecture and reusing these results as an independent test, or declaring a
model-family ceiling. Keep the run as a diagnostic. Next prioritize an approved
task with grounded decision labels and a reason the required modalities are
complementary; separately measure task quality and modality necessity. The
95%+ product objective and multi-modality requirement remain unchanged.

## Artifacts and hashes

External run directory:
`C:/CodexArtifacts/pretrained-reuse-r2/clevrer-textfree-temporal-readout-v1-attempt-01/`.
The run-report SHA-256 is
`ba2cc29125e7e1c0ea3300c6ba835d26d8295a2a9674267cb01dc77e581e3bfd`.

| Artifact | Mean MLP SHA-256 | BiLSTM SHA-256 |
|---|---|---|
| Best checkpoint | `c19c2602b2d1e1938fe4031c4101518d443be9fd3e788bc4d2cf25f324accd45` | `7aaf177147d4b18a25577cf887b08f7fbbae62a255591a2d63551faaf7430d6c` |
| Final checkpoint | `2b70326b56387369d5b4ef636ec7bfd86e246c1d1fc599b22807180194db3b24` | `b80a2702a3da300e3d51895142eb35174fc7928cdeb323ceaea2c7bf88a24b5c` |
| Training history | `5d85b5d925bd9fb2f98cac15d79b84e85ae62e095b5420bf1ab75beae9c3e97f` | `d48443b400ed9268b4406da88e3c30e3886f85a17aef2ca7425fe3e15af8f416` |
| Selected validation predictions | `8a60bd6979fd780cb6380a621cd1b8164a84c9f6e9c27d54d50a23a062282eae` | `eebbefcd8b453fa8542f45dbd29a939935077a6cab5ea35c3e140473257561a1` |

The run report and every checkpoint, history, and prediction file were
re-hashed after completion. Each prediction file contains all 321 validation
IDs exactly once with normalized option probabilities. No weights or media are
committed to Git.
