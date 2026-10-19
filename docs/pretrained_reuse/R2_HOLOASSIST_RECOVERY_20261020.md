# HoloAssist unused-train recovery and temporal corpus (2026-10-20)

This continuation keeps the existing 367-session pilot and its artifacts
unchanged. It freezes a larger train/validation selection from unused source
records and prepares a separate text-free causal action-quality experiment.
The official test split, labels, and media were not opened.

## Recovered media and split integrity

The immutable selection has SHA-256
`66f578b89ddb1757477b6c75e31b9d3fb7320d6832ae3d23dcd72ad67be17498` and
contains 1,042 train and 264 validation sessions (1,306 total). Selected media
totals 117,074,880,152 bytes. All 1,306 files were downloaded and checked
against the frozen media manifest; there are no partial downloads. The media
manifest SHA-256 is
`f1944b66510897fdcf09ec5cfb2b57ca9601eae11191d28b7ae3b6766f0539ed`.

The previous validator expected the archive ETag to be embedded in
`selection.json`, but the actual frozen selection has no such field: the
downloader pins the ETag separately. The validator now requires an explicit
`--archive-etag`, matches it to the download receipt and every file receipt,
and reports the maximum accepted stream-duration tolerance. This corrects the
schema assumption without weakening selection, content-hash, split, codec,
channel, sample-rate, or media checks.

One source recording's audio endpoint differed from its video endpoint by
slightly more than two video-frame intervals due to AAC packet granularity.
The timing gate now admits two measured video-frame intervals plus one AAC
packet interval (`frame_size / sample_rate`), and records both quantities.
The observed maximum audio/video endpoint difference is 0.0689127 seconds; the
maximum allowed tolerance is recorded in
`av-validation-report-v2.json`. The report passed with 1,042 train and 264
validation media rows. Its SHA-256 is
`fbc6fced29f7a87b9bdc5fbb3c64a18b3f140ae5058eb36db80c5f27d8cc85db`.
This is a media-timeline validation tolerance, not a model-quality threshold.

The corpus lives outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/holoassist-textfree-unused-train-recovery-02/`.
The frozen official source split archive SHA-256 is
`e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621`; the
train/validation annotation SHA-256 is
`cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c`.

## Action events and causal windows

The source reuses some action-event IDs within one record, including IDs
colliding with other event categories. Dropping duplicate IDs would silently
lose valid supervision. The event builder therefore forms stable unique IDs as
`taskId:sourceEventId:sourcePosition` while retaining the original source ID
and position.

The action corpus receipt SHA-256 is
`ccdcab838d8afca512c720ce221a6567bd07b38546238293af7e9ba02d9acdf0`. It
contains 88,762 train and 22,761 validation action events from the same
1,042/264 disjoint sessions. Incorrect-action prevalence is 4.79% in train and
5.11% in validation. Events outside measured media intervals, reversed
intervals, and non-action categories are counted in the receipt and are not
truncated or relabeled.

Thirty-second causal windows retain unlabeled windows as recurrent context,
while excluding them from loss and metrics. The window receipt SHA-256 is
`7e4862bae7701ccb8eaf4958e4efaa8079dad291dc094b2dec9f66dfa0b31475`.

| Split | Sessions | Windows | Labeled | Positive | Negative | Unlabeled context |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,042 | 9,763 | 9,351 | 2,652 | 6,699 | 412 |
| Validation | 264 | 2,498 | 2,373 | 719 | 1,654 | 125 |

The ordered train/validation window JSONL hashes are
`406b4e6cf18588efb931809ac74415609b6b17500dd1ac02225df48ca9514b24` and
`584f6e9dfc5b5590f5108d954064f3ad05745f77f5e4211c65a1e0f9aaeaf149`.
The action-event JSONL hashes are
`cb00318d894dcf9bc05e146132794434af127b7018f10bbb6fd8b55ba9e01829` and
`7f98cc593e8b17521e13b351e58c0dd54c952a62bba81fe1ccda0989b4eafeef`.

## VideoPrism execution readiness

The pre-existing pinned CPU/JAX environment was reused unchanged: Python 3.11,
JAX/JAXlib 0.6.2, TensorFlow 2.19.1, PyAV 18.1.0, NumPy 2.1.3, and psutil
7.2.2. The VideoPrism source checkout is at pinned commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8` with no tracked modifications. The
local 457,475,296-byte checkpoint hashes to the pinned SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.

A separate-process real-media smoke encoded one eight-frame window to a finite
`[1, 768]` feature vector on `TFRT_CPU_0`. It loaded 114,365,184 parameters,
peaked at 4,207,423,488 bytes RSS, and took 9.73 seconds including cold model
load/JIT compilation. This single-window cold timing is not extrapolated as
steady-state throughput. At the time of the smoke, PID 19516 was an active
Orbitune training process using 15,821 MiB on the local 16 GiB RTX 3080 Laptop.
No CUDA extraction or training was started. Full feature extraction and all
readout fitting/evaluation remain pending; no HoloAssist quality result is
claimed by this recovery note.

## Reproduction and tests

The source split, annotations, media, feature banks, and readout checkpoints
remain external to Git and were not overwritten. The updated tests cover the
actual selection schema, explicit ETag pin, AAC packet timing allowance, and
stable disambiguation of repeated upstream event IDs.

## Ongoing frozen extraction and AV intervention follow-up

The pinned VideoPrism extractor is running as attempt-03 under the cooperative
HotUGPU reservation. At 2026-10-20 05:33 JST it had saved 2,048 / 12,261
development-window embeddings; all completed rows are finite. The progress NPZ
is 5,953,637 bytes and its embedded fingerprint binds the frozen selection,
corpus receipts, VideoPrism revision, checkpoint hash, preprocessing, and
extractor source. An auxiliary `progress-fingerprint.json` still says 192 rows
and is stale; it is not used as the progress authority. The completed-row count
comes from the NPZ `completed` mask. A read-only integrity check matched every
stored sample ID and its order to the frozen train+validation JSONLs and matched
the recorded ordered-ID hash; the 2,048 completed flags form a contiguous
prefix. Completed vectors are finite, nonzero, and nonconstant (L2 norm
15.97–16.70 in this checkpoint).

The extraction remains on the frozen CPU/JAX path. The extractor itself sets
`JAX_PLATFORMS=cpu` and hides CUDA for VideoPrism; this is not a GPU job even
though HotUGPU reserves 6,000 MiB (6,169 MiB observed in use) and the RTX 3080
reports 0% utilization. The same guard, extractor, post-extraction continuation,
and two evaluation watchers remain alive; none has been restarted or altered.
Across the saved intervals 1,920→1,984 and 1,984→2,048, throughput was about
24 embeddings/minute (roughly 2.5 seconds per window). A naive projection at
that short-run rate leaves about 7.0 hours for the remaining rows; this is a
planning estimate, not a runtime guarantee. Audio extraction, cache packing,
and the three matched readout arms have not started yet; the existing
continuation waits for the completed atomic video bank.

The readout's joint arm is followed by a separate fixed-validation intervention
evaluator, `scripts/evaluate_holoassist_temporal_interventions.py`. It checks
the completed run, exact train/validation feature hashes, receipt, checkpoint
hash, fixed task vocabulary and all three arm results before loading only the
joint best checkpoint. It compares baseline replay, missing audio, missing
video, within-session audio time reversal, within-session video time reversal,
and audio reassignment between validation sessions matched on task type and
window count. It records mean video/audio gate weights overall and per task as
an auxiliary diagnostic. It writes its own six per-example prediction files
and metrics; the baseline must replay the saved probabilities within 1e-6.
Sealed audit and official test data are not loaded. A dedicated watcher waits
for the existing pipeline to finish and then runs this evaluation through
HotUGPU in a new output directory; it does not overlap the three-arm fitting
run.

A second watcher waits for the intervention evaluator and then computes seven
paired session-cluster bootstrap comparisons (joint vs each unimodal arm, plus
joint vs each fixed intervention), with 10,000 replicates and seed 1701. Each
report records the paired prediction hash and percentile interval. These are
validation-only diagnostic intervals; because checkpoint selection used the
same validation sessions, they are not independent confirmation and cannot
support a product promotion claim.

CPU tests cover modality-mask behavior and immutability, deterministic
session-level interventions, fixed sample-ID coverage, replay parity against
the training evaluator, and receipt/hash-bound loading. The full CPU suite
passed (2,487 passed, 1 skipped); Ruff lint passed and the four changed Python
files are formatted. `ruff format --check .` still reports 379 pre-existing
unformatted repository files, which were not bulk-reformatted. These code and
runtime checks do not constitute HoloAssist quality results. This task-specific
AV study alone does not close the broader R2-M matrix or R0-R8 quality gates.
