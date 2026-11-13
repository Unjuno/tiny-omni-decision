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

1. Resolve the remaining visual identity risk against already-used non-sealed
   Clevr-4 image assets. If this check cannot pass without opening sealed data,
   keep CATER unadmitted and select a different admissible task.
2. Re-run a CPU-only frame-decoding smoke against a healthy clip, one damaged
   clip with an intact selected PTS, and a sample with a missing selected PTS.
   Confirm exact `[13,256,768]` finite VideoPrism output after preprocessing,
   and fail closed for the unavailable sample.
3. Only if source identity passes, create a fresh extraction attempt from the
   versioned 4,360-train / 1,120-validation manifest. Keep the 1,248-file
   partial attempt intact; reuse it only through an explicit hash-checked
   migration receipt, otherwise re-extract to a new directory. Do not resume
   the legacy attempt using changed decoder code.
4. Fit the fixed rank-free readout on seed 17 only. Encoders remain frozen; only
   the approximately 0.51M-parameter option-conditioned spatial-attention +
   GRU + scorer head is trainable. Compare the same head with full video input
   against a separately trained text-only condition.
5. Select checkpoints by macro validation NLL across the four fixed intervals,
   with the scripted early-stopping rule. Save per-interval Accuracy/NLL/Brier/
   ECE, all validation predictions, train/validation metrics, and scene-group
   paired bootstrap. Run static-frame, interval reversal, interior shuffle,
   and video-permutation controls on the selected video checkpoint.
6. Keep the result development-only. Do not infer four-modality quality,
   product readiness, or 95–98% performance from this synthetic video task.

The two encoders plus this readout total 132,265,729 unquantized parameters
(VideoPrism 114,365,184 + MiniLM 17,389,824 + readout head 510,721). Packaged
size, quantized size, and runtime latency have not been measured here.

The local MiniLM-L3 readout input was rechecked before the full run: its
`model.safetensors` SHA-256 matches the pinned
`cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`; the local
configuration reports a 384-wide, 3-layer encoder. The readout report now also
records a sorted file/size/SHA-256 manifest for all model, tokenizer, and
configuration files instead of recording only the weight hash.

## 2026-11-14 extraction and packet-integrity correction

The interval-local bulk extraction attempt did **not** complete. Its stdout
reached 1,248 of 5,486 samples (train rows first); `run.json` was left with the
stale value `RUNNING`, no `result.json` exists, and the process is terminal.
The attempt directory is preserved with 1,248 atomic feature files totaling
6,382,826,776 bytes. The extractor failed at
`cater-003291-disp-075-150` with PyAV `InvalidDataError` while decoding packet
PTS 90. This is extraction plumbing evidence only; no readout fit or quality
evaluation ran.

The earlier `cater-media-frame-audit-20261113-attempt-01/records.jsonl` does
not contain `decoded_frames` for `CATER_new_003291.avi`; its `stream_frames`
value is container metadata, not proof of a successful full decode. A new
full-media audit used PyAV 18.1.0 / libavcodec 62.28.102 and rechecked all
1,504 media hashes against the pinned media manifest before packet decoding.
It found six train videos with 54 unreadable MJPEG packets (`InvalidDataError`);
all six local byte hashes match their exact manifest entries. Validation has
no decode-error videos. The audit took 187.07 seconds and its record SHA-256 is
`9ecbd1cb4361d7c9d2a5f0ada6703f484d4c5b18f070335dd609401fefc393ae` at
`D:\CodexArtifacts\pretrained-reuse-r2\cater-media-frame-audit-20261114-attempt-01\records.jsonl`.

The extractor now maps MJPEG packets by packet PTS, skips only invalid packets
outside the 13 selected timestamps, and fails closed if any selected timestamp
is unreadable. A CPU integration check over all 24 intervals from the six
affected clips decoded exactly 18; six intervals contain at least one missing
selected frame. No replacement frame, interpolation, or altered frame schedule
was used. The derived manifest keeps 4,360 train examples / 1,196 train video
groups and all 1,120 validation examples / 308 validation groups, with zero
video overlap, zero normalized-content overlap, and unchanged 5–10 option
counts. Its train SHA-256 is
`23ccef3c1eebf90bafb6726d1933c8105b0acf96f6c02d581d50bc95c56de2e8`; validation
is unchanged at `dd9e7920026228d1e8442aea45883c36f1def7f54e96dd126856165b20079fa0`.
The 12 train rows removed by the deterministic media gate comprise six
out-of-range questions from the two short videos and six questions whose
selected frame packets are unreadable. The exact IDs and missing PTS are in
`D:\CodexArtifacts\pretrained-reuse-r2\cater-temporal-decision-manifest-20261113-attempt-03\filter-summary.json`.

Focused CPU tests (23) and Ruff pass for packet audit, manifest filtering, and
frame extraction. The original 1,248 feature files were not reused or changed.
Before any large extraction or readout fit, CATER still needs a bounded visual
identity check against already-used Clevr-4 assets; the task remains
`REVIEW_NOT_ADMITTED`. A future run must use the 4,360-row derived train
manifest, a fresh output path, and the code-hash-aware run metadata. It must
not report the old attempt as complete or silently resume it under changed
decoder code.

## 2026-11-14 cross-source visual identity gate and fresh extractor smoke

The bounded visual identity screen is complete using the 4,360-row train and
1,120-row selector manifests and only the previously used, non-sealed Clevr-4
train/selector/reporting images. It compared 67,480 fixed CATER interval-frame
packets with 10,531 Clevr-4 images. There were zero exact canonical RGB64
matches. Per-CATER-frame nearest pHash distance had minimum 0, median 10, and
174 pairs at distance <=2; all 174 were inspected in contact sheets. No pair
was visually identified as the same media or a near duplicate. The two exact
pHash matches and the low-distance pairs share the renderer/background style,
but differ in rendered objects, geometry, count, or arrangement.

Because every selected CATER video has a numeric suffix represented among
previously used Clevr-4 image IDs, a separate direct same-suffix comparison
covered all 67,480 frame/image pairs. It also found zero exact canonical RGB64
matches; pHash minimum/median/p95 were 2/28/36. The 48 nearest same-suffix
pairs were visually inspected, including all 12 pairs at pHash distance <=8;
none was identified as a duplicate. This resolves the bounded source-identity
gate for this local development candidate, but pHash and finite visual review
cannot prove absence of every possible transformed near duplicate.

The machine-readable review receipt is
`D:\CodexArtifacts\pretrained-reuse-r2\cater-clevr4-visual-identity-audit-20261114-attempt-02\visual-review.json`
(SHA-256 `e4c906fb758f8678cb1e8bc239e36d92962f9a9c99e660b01bfeaa18f5d83499`).
It records hashes for the comparison summaries, pair lists, contact sheets,
and cached input hashes. No media was modified, no GPU was used, and no sealed
audit or official CATER validation/test list was opened.

The corrected packet-aware extractor then passed a fresh two-example CPU smoke
using one train and one selector interval. It verified the exact pinned
VideoPrism source and checkpoint, selected the 13 frame PTS within each
question interval, and wrote finite `[13,256,768]` float16 features. Result:
`D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-packet-aware-smoke-20261114-attempt-01\result.json`
(SHA-256 `375b3f0688fdfd7c5418eedbecaa1fa13f6a66363860e2d9a3786c9265e524`).
The run took 14.52 seconds, used the CPU
backend, and recorded about 1.90 GB final process RSS. The source commit was
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; checkpoint SHA-256 remains
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.

**Disposition:** admit CATER only for this local, video-plus-text-conditioned
Path A development readout. The readout embeds the question and candidate
descriptions with MiniLM, so this experiment is not the roadmap's direct
text-free video-only M0/M1 probe. Keep the task, manifests, intervals, option
sets, split, and encoders frozen. This does not establish general video quality,
multimodal benefit, an independent audit, or product readiness. The failed
1,248-feature attempt remains preserved and unusable; the bulk run must use a
new output directory. After extraction, fit the already specified fixed readout
once and run its preregistered temporal/order/permutation controls. Bootstrap by
video scene.

The new full extraction attempt was launched at 2026-11-14 02:26:44 JST in
`D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-interval-features-20261114-attempt-01`
with the exact derived manifests, pinned source/checkpoint, CPU JAX backend,
and a fresh output directory. Launcher PID `16428` and Python worker PID
`20720` were confirmed live at 02:27 JST; both were BelowNormal priority. The
launch receipt is
`D:\CodexArtifacts\pretrained-reuse-r2\cater-videoprism-interval-features-20261114-attempt-01-launch-provenance.json`.
At 02:31 JST, `run.json` reported `RUNNING`; 51 train feature files had been
written (260,836,698 bytes), with validation extraction not yet started. The
worker remained alive at BelowNormal priority. `nvidia-smi` continued to show
only the unrelated Orbitune process on the RTX 3080; this extraction used no
GPU. These are progress counters, not a completion or quality result.

Before fitting the readout, a code-path check found the full-cohort guard still
expected the earlier provisional 4,372 train queries. The frozen derived
manifest actually has 4,360 train rows / 1,196 scenes and 1,120 selector rows /
308 scenes, with hashes matching the launch receipt. Updated the guard to the
frozen 4,360/1,120 counts and added a regression test. The readout tests passed
(6 passed) in the existing torch environment and Ruff passed. This fixes a
pre-existing evaluation blocker; it does not change the extraction process,
data, model, or training conditions.

At 03:36:39 JST, the new packet-aware bulk extractor successfully wrote the
exact interval feature that stopped the prior decoder attempt:
`cater-003291-disp-075-150`. Its selected packet PTS are
`[75,81,88,94,100,106,112,119,125,131,138,144,150]`; the unreadable PTS 90 is
not selected. The saved tensor is finite `[13,256,768]` float16 and its payload
SHA-256 matches the embedded record. This verifies the specific prior failure
case without changing manifests or decoding policy. At 03:38:46 JST the full
run remained `RUNNING` at 1,280/5,480 samples (1,290 atomic feature files); no
readout or quality evaluation has run.
