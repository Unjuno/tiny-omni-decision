# R2-M RMS-window participant CV results (2026-11-21)

## Execution and integrity

The preregistered RMS-window comparison completed all seven participant folds
and all 14 Audio-only / Audio+sensor fits. The run used the local RTX 3080
Laptop GPU, CUDA 12.4, PyTorch 2.6.0+cu124, Python 3.11.9, seed 17, 80 epochs,
batch size 32, AdamW at 0.001 with weight decay 0.1, and no early stopping.
Elapsed time was 3,756.6 seconds (62.6 minutes); peak allocated VRAM was
183,261,696 bytes. No cloud compute was used.

The run output is
`C:\CodexArtifacts\pretrained-reuse-r2m\r2m-window-rms-participant-cv-v1-attempt-01`.
It contains 5,449 unique TRAIN rows per arm and 5,449 paired OOF predictions.
The feature cache is TRAIN-only, with index SHA-256
`e3121bcbd84cfd9d86da20971b99591713ccb59966dfcb90a322a9bc3db3f16a`.
Effective config SHA-256 is `c2f339bdb0f8873f5e1fe739d3c60a51d9a00889280482e01c9eb0aa662ed3e0`;
the config's self-excluding digest is
`94f872b096b50e3ec3dfd4de11d4612e479b35e6658723f6edbae05586cef972`. Source
commit was `88f43b8fd214fe8a1d09e4543b1d0ec271598e67`. The sealed audit was not
loaded, and the source split's calibration/validation signal rows were not
included in the model cache or fit. All seven fold plans match the legacy CV.
All 14 best and final checkpoint hashes, training-history hashes, output
hashes, sample identities, and normalized finite probabilities passed an
independent integrity check.

## Pooled OOF results

All runs use the same 5,449 sample IDs and 24 supplied activity choices.
Metrics are from grouped out-of-fold predictions over the seven participants
in the original TRAIN pool; they are development evidence, not a new blind
test.

| Run / arm | Accuracy | Macro accuracy | NLL | Macro NLL | Brier | Macro Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy point features, Audio-only | 79.43% | 80.12% | 0.7053 | 0.7044 | 0.3060 | 0.3020 | 0.0851 |
| Legacy point features, Audio+sensor | 72.29% | 73.27% | 1.2269 | 1.2633 | 0.4237 | 0.4227 | 0.1178 |
| RMS run, Audio-only | 79.26% | 79.19% | 0.8008 | 0.8373 | 0.3121 | 0.3170 | 0.0964 |
| RMS run, Audio+sensor | 78.77% | 79.27% | 0.9093 | 1.0043 | 0.3359 | 0.3305 | 0.1110 |

Within the RMS run, fusion changed pooled Accuracy by −0.50 percentage points
and macro Accuracy by +0.08 points relative to Audio-only. Macro NLL worsened
by 0.1670, macro Brier by 0.0135, and ECE by 0.0146. Fusion Accuracy improved
for 2 of 7 held-out participants and regressed for 5. The preregistered
criterion is not met: RMS fusion does not improve both macro Accuracy and
macro NLL over Audio-only, and its proper scores deteriorate.

Compared with the legacy point-sampled fusion run, RMS fusion has +6.48 points
pooled Accuracy, +6.00 points macro Accuracy, and lower pooled NLL (−0.3177),
macro NLL (−0.2590), macro Brier (−0.0922), and ECE (−0.0068). This is a
promising representation-path signal, but not sufficient to retain this
source/task fusion candidate given the same-run Audio-only comparison.

An exploratory paired participant-cluster bootstrap repeated the previous
diagnostic: 10,000 PCG64 resamples of the seven participant clusters, seed 17,
equal weight per participant. RMS fusion minus Audio-only mean participant
Accuracy was −0.92 percentage points (95% percentile interval −3.14 to
+1.50); NLL was +0.0889 (+0.0768 to +0.2004); Brier was +0.0289
(+0.0075 to +0.0634). With seven clusters and overlapping fit participants,
these intervals are descriptive only.

## Reproducibility caveat and decision

The old and new Audio-only OOF predictions cover identical IDs, targets,
participants, fold plan, model settings, and audio feature values, yet the
probability vectors were not reproduced exactly: 746 of 5,449 decisions
changed. Both effective configs record the same Python, PyTorch, CUDA, GPU,
seed, fold plan, and optimizer schedule. The execution source hashes recorded
for the legacy run match the checked-in CV/trainer sources at commit
`765ceb21`; the current changes add cache-scope checks and final-fold metrics,
not a deliberate Audio-only training change. The cause of the run-to-run
variation remains unknown, so cross-run legacy deltas are contextual rather
than a strict paired causal estimate. The within-run RMS Audio-only versus
Audio+sensor comparison remains paired and determines the preregistered
decision.

**Decision:** retire R2-M audio-vibration fusion for this supplied-activity
task. The RMS envelope improved on the old fusion result but did not beat the
same-run Audio-only model on the preregistered proper-score criterion. Do not
increase capacity, add epochs, or tune this source further. Keep the sealed
audit unopened and retain all existing legacy artifacts unchanged. This
conclusion is specific to this source, task, and seven-participant development
CV; it is not evidence against multimodal fusion generally.
