# CATER temporal frozen-feature Path A preflight — 2026-11-13

## Scope and current state

This is a development-only video task for checking whether pretrained frozen
VideoPrism representations can support option-conditioned temporal decisions.
It is not a general video benchmark, a multimodal result, or evidence for the
project's 95–98% quality aspiration.

The fixed task asks which scene object has the greatest annotated net
displacement between the endpoints of one 75-frame interval. It retains all
5–10 object options. Only the frozen train/selector manifests are used; official
CATER validation/test lists and any sealed audit are not loaded.

## Data acquisition and split integrity

The source task manifests are frozen at:

- train: 4,372 question rows / 1,196 video groups, SHA-256
  `c8e6c911a3edefdc3bfb71114213011f243a9fe66ea7e1d772fbe2ff93dad217`;
- selector: 1,120 question rows / 308 video groups, SHA-256
  `dd9e7920026228d1e8442aea45883c36f1def7f54e96dd126856165b20079fa0`.

All 1,504 selected AVI assets were acquired by verified HTTP byte ranges from
the official archive. The 14.2-GB ZIP itself was not downloaded. The archive
central-directory tail matched pinned SHA-256
`6e07fd7f329227e8f63c1638cecac301bc373bc1443693d48cbd47211721bc01`. The fetch
receipt reports 1,196 train videos, 308 selector videos, zero video-group
overlap, and 4,676,061,576 uncompressed media bytes. The media manifest SHA-256
is `13cdb158d79fafb62f1e71713d1f24c7a76b5515b47f3a3ce0e6d626fd897673`; the
fetch-report SHA-256 is
`9f6348b067b21bb14b70b6305cabc3e2aabda97afdd4c3f889e7189708eb699e`.

The first range-fetch attempt ended with provider HTTP 503 after writing a
partial directory. That directory was preserved. Resume verified each
existing file against the pinned archive member size and CRC before fetching
only missing files; the completed media manifest hashes every asset. No media
was added to Git.

The train/selector task manifests have zero overlap of normalized question plus
sorted option-ID content (4,372 and 1,120 unique contents respectively). Scene
grouping and conservative Clevr-4 numeric-ID exclusions are described in the
[metadata feasibility audit](R2_CATER_TEMPORAL_TASK_FEASIBILITY_20261113.md).
That prior coarse attribute audit found no eligible CATER scene matching the
Clevr-4 image-wide uniform shape/color/texture signature. This does not rule out
arbitrary visual near-duplicates.

## Train-only visual sanity check

Before fitting a readout, five train videos were selected deterministically
with seed 17 and visually inspected as 13-frame contact sheets. All decoded to
301 frames, the sampled shape/color patterns were distinguishable, and the
clips visibly contain object movement. IDs, media hashes, eligible intervals,
contact-sheet hashes, and the selection seed are in the external receipt:

`D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-visual-audit-20261113-attempt-01\receipt.json`

The five videos are `CATER_new_002903`, `003021`, `003590`, `004160`, and
`005161`. This small train-only inspection is a media sanity check, not a
quantified cross-dataset near-duplicate search or a label-quality audit of every
scene. CATER remains `REVIEW_NOT_ADMITTED` until the remaining source-identity
review is resolved.

## Pinned representation and extraction smoke

VideoPrism-B is pinned to model revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, source commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, and
114,365,184 parameters. The earlier CPU smoke exercised a 13-frame input.
The first extractor version sampled `0, 25, …, 300` from each full video and
stored `[13, 256, 768]` float16 arrays. That design is rejected for interval
evaluation: temporal tokens may carry later-frame context into earlier tokens,
so full-video features could expose future content to a 75-frame question.

The two-video smoke passed with one train video and one selector video:

- output: `D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-temporal-smoke-20261113-attempt-03`;
- result SHA-256: `5db5be68ca5dbe0a4563a95fddf00bec88b2e54cb99b94f5009e3b46befd5c7a`;
- both features were finite and had shape `[13,256,768]`. This only confirmed
  the old tensor path; it is not an admissible task feature cache and was not
  used for quality evaluation.

A separate one-group, one-epoch Decision readout smoke passed for both a
video-conditioned head and an independently trained text-only control. It
verified prediction serialization, validation metrics, and grouped bootstrap
plumbing on one selector question. Its quality numbers are not interpretable.
The output is retained at
`D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-readout-smoke-20261113-attempt-03`.

## Interrupted extraction and media correction

The first full-video extraction stopped after 192 of 1,504 video feature files.
It failed on `CATER_new_005781.avi`, which decodes to 105 frames;
`CATER_new_005780.avi` decodes to 100 frames. An independent inventory found
1,502 videos with 301 decoded frames and these two short train videos. They
contain only the first 75-frame task window. Six out-of-range questions were
removed by a deterministic media-based filter while preserving each video's
observable 0–75 question. The derived train manifest contains 4,366 rows / 1,196
video groups, SHA-256
`679c4a2d1fb53942e68c3433d0590e5238dbb947f5ec0f3f80548a3dbf0021b2`;
validation is unchanged at 1,120 rows / 308 groups, SHA-256
`dd9e7920026228d1e8442aea45883c36f1def7f54e96dd126856165b20079fa0`.

The old run directory remains preserved but cannot be resumed for task scoring.
Its features encode whole videos and are invalid for the interval task because
of potential temporal leakage:

`D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-temporal-features-20261113-attempt-01`

Launch provenance, including project base commit and extractor SHA-256, is
stored beside that directory as
`D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-temporal-features-20261113-attempt-01-launch-provenance.json`.
The first three interval throughput observations vary substantially, so the
remaining runtime estimate is provisional. Each feature file is written
atomically and independently; resume checks the run inputs and feature hashes.
No Decision quality result or admissible full-corpus feature cache is
available.

## Corrected interval feature path

The extractor and readout now key cached features by sample ID and media hash.
Each question interval is decoded and encoded independently using 13 evenly
spaced frames including its endpoints. This preserves the encoder's 13-frame
input shape while ensuring no frame outside the labelled interval is supplied.
The readout uses local frame positions 0–12; the permuted-video control pairs
the same interval from another validation scene. Short videos retain only
questions whose complete interval exists in the media.

Focused CPU regression tests pass (34 tests) and Ruff passes. The pinned source
checkout and runtime were present at the original C: path; only the local model
weight file was missing. The exact pinned checkpoint was reacquired into a new
external artifact directory and its SHA-256 verified:

`D:\CodexArtifacts\pretrained-reuse-r2\videoprism-checkpoint-20261113-attempt-01\flax_base_f16r288_repeated.npz`

SHA-256 is `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
(457,475,296 bytes); source commit remains
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.

The corrected extractor passed two CPU smoke runs, each with one train and one
validation sample. The 0–75 run result SHA-256 is
`f823fc8f429cb59f1960e861004e630f8836589f5b3b9cc2916486fa272a54a9`; the
75–150 run result SHA-256 is
`9cce148f06b452fe1a0f7b3bb34dac6721570110068494c215c429347ffeccb7`. Both
produced finite `[13,256,768]` features, and their recorded sampled frame
indices stayed within the corresponding inclusive intervals. The first run's
peak recorded RSS was 1,907,175,424 bytes. A one-epoch, one-video-per-split
readout wiring smoke also completed and saved its report at
`D:\CodexArtifacts\pretrained-reuse-r2\cater-interval-readout-smoke-20261113-attempt-01\report.json`
(SHA-256
`c66dd838bee5b7d751588c7341e3ba0df1d6483ecbc244803ca0bfa1748276978`);
its one-example metrics are not quality evidence.

The full interval-local extraction started at 2026-11-13 23:28 JST as CPU-only
process PID 47796 at BelowNormal priority. It processes 4,366 train and 1,120
validation sample IDs from the frozen corrected manifests, with 1,196 and 308
video groups respectively. The new output path is
`D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-interval-features-20261113-attempt-01`;
launch provenance is stored beside it. No quality fitting or metric-based
selection is part of extraction. Another Orbitune process occupies the RTX
3080; the smoke inference and readout checks were CPU-only. D: had 2.73 TiB
free before launch. Measured steady-state CPU forward time was 2.86–2.96
seconds per sample after first-process compilation (10.94–11.65 seconds for
each cold first forward). The naive model-forward projection is about 4.44
hours for 5,486 question intervals, excluding media decode, file I/O and later
readout; this projection is based on four smoke forwards and is not a runtime
guarantee.

## Next gate

1. Restore or reacquire the exact pinned VideoPrism source/runtime without
   changing its checkpoint; run an interval-only CPU smoke before bulk
   extraction.
2. Extract one feature per sample ID (4,366 train + 1,120 validation), verifying
   interval-local frame indices, media hashes, split IDs and finite
   `[13,256,768]` shape. Write to a new attempt directory; do not resume the
   invalid full-video attempt.
3. Fit the fixed rank-free readout on seed 17 only. Encoders remain frozen; only
   the approximately 0.51M-parameter option-conditioned spatial-attention +
   GRU + scorer head is trainable. Compare the same head with full video input
   against a separately trained text-only condition.
4. Select checkpoints by macro validation NLL across the four fixed intervals,
   with the scripted early-stopping rule. Save per-interval Accuracy/NLL/Brier/
   ECE, all validation predictions, train/validation metrics, and scene-group
   paired bootstrap. Run static-frame, interval reversal, interior shuffle,
   and video-permutation controls on the selected video checkpoint.
5. Keep the result development-only. Do not infer four-modality quality,
   product readiness, or 95–98% performance from this synthetic video task.

The two encoders plus this readout total 132,265,729 unquantized parameters
(VideoPrism 114,365,184 + MiniLM 17,389,824 + readout head 510,721). Packaged
size, quantized size, and runtime latency have not been measured here.
