# AVATAR encoder runtime smokes (2026-11-01)

These are bounded execution checks for the frozen encoders on three fixed
validation examples. They are not quality measurements, modality controls,
representative throughput benchmarks, or evidence of a blind evaluation. The
AVATAR audit annotations were not opened. No fitting or checkpoint updates were
performed.

## Inputs and pinned weights

| Encoder | Revision / source | Parameters | Checkpoint SHA-256 | Device |
|---|---|---:|---|---|
| EfficientAT DyMN-0.4 | `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806` | 1,961,239 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | RTX 3080 Laptop, CUDA 12.4 |
| VideoPrism-B | `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`; source `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` | 114,365,184 | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | JAX CPU (`TFRT_CPU_0`) |

The combined encoder count is 116,326,423 parameters, below the roadmap's
provisional 200M resident-parameter ceiling. This says nothing about the
eventual serialized package-size target.

## Three-example smoke results

The same three validation rows (0, 100, 500; source groups `-EFu7tP-QEk`,
`0OtOrR70M1A`, `9o2GcFviRBc`) were used for the encoder smokes.

| Measurement | Example 1 | Example 2 | Example 3 | Mean |
|---|---:|---:|---:|---:|
| PyAV open/decode/preprocess of one target audio plus four candidate videos | 0.2092 s | 0.5041 s | 0.2745 s | 0.3293 s |
| VideoPrism video decode and preprocessing | 0.7490 s | 0.2944 s | 0.3251 s | 0.4562 s |
| EfficientAT forward (including first-call warm-up) | 0.9343 s | 0.1408 s | 0.1373 s | 0.4042 s |

EfficientAT strictly loaded the pinned state dict and produced finite `[1,384]`
features from 64,000-sample, 32 kHz windows. The model's peak CUDA allocation
was 19,929,600 bytes (reserved 33,554,432 bytes). This was a one-batch smoke,
not a throughput estimate for bulk extraction.

VideoPrism produced finite `[3,2048,768]` tokens and `[3,768]` pooled features
for eight frames per clip. The task-smoke preprocessing sampled eight frames
over a one-second interval centered at the annotation timestamp, center-cropped
to square, and resized to 288 using JAX bicubic. The maximum observed frame
timestamp error was 0.039 seconds. This is the implemented smoke preprocessing;
it is not asserted to be the official upstream AVATAR preprocessing.

Separate batch-throughput probes with the same pinned VideoPrism model and
input shape, using synthetic arrays rather than media, measured:

| Batch | Reused forward times | Mean | Mean per item |
|---:|---|---:|---:|
| 3 | 3.9004, 3.7142, 3.9440 s | 3.8528 s | 1.2843 s |
| 8 | 9.7349, 9.2244, 9.3123 s | 9.4239 s | 1.1780 s |

Batch 8 had peak process RSS 4,154,802,176 bytes. At these small samples,
batching improved measured per-item model time by about 8%; variability and the
synthetic inputs limit the precision of this estimate.

## Runtime implication and limits

There are 13,688 unique target/candidate video assets across the frozen
train/validation choice sets. Applying the batch-8 synthetic throughput and
three-video decode/preprocess mean gives a rough VideoPrism extraction estimate
of 4.47 hours of model work plus 1.73 hours of decode/preprocessing, or about
6.2 hours total. This is a planning estimate from three real media examples and
three repeated synthetic batches, not a measured corpus run; I/O contention,
JAX scheduling, failures, and preprocessing details may change it materially.

At the time of the smokes, the RTX 3080 had 15,757 MiB used and about 419 MiB
free, with the existing Orbitune process tree still alive. Do not start another
GPU job while that guard remains occupied. The EfficientAT smoke's small model
allocation does not imply that the remaining GPU memory is available. No cloud
compute or cost was used.

The machine-local output receipts are:

- `C:/CodexArtifacts/pretrained-reuse-r2/avatar-av-decision-20261101-attempt-02/efficientat-validation-smoke.json`
- `C:/CodexArtifacts/pretrained-reuse-r2/avatar-av-decision-20261101-attempt-02/videoprism-validation-smoke.json`

Next, complete media integrity and path-identity checks for the frozen
train/validation media, then construct separately hashed frozen feature caches
and run option-only, audio-only, video-only, joint, missing-stream and
within-split shuffle controls before any readout fitting. Preserve the sealed
audit as unopened.

The full archive-level media container pass has since completed: all 3,009
unique clips referenced by train/validation choices opened with readable audio
and video streams, and no referenced annotation timestamp exceeded its media
container duration. The receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/avatar-av-decision-20261101-attempt-02/av-media-integrity-01.json`
(SHA-256 `ab117d7b34f4de1056d124d88634a93c7bc8ef82f5994fda21137e06f83ec00a`).
Full media payload decoding and precise feature-window alignment remain
sampled checks rather than a corpus-wide decode.
