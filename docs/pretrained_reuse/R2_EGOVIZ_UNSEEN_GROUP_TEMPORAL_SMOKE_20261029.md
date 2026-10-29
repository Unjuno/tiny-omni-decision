# EgoViz unseen-session temporal smoke — 2026-10-29

## Scope and outcome

This is a raw-media, synchronized-window, and frozen-feature pipeline smoke for
the frozen unseen-session plan. It is **not** a model-quality evaluation: no
readout was fit and no sealed-audit media or labels were accessed. The fixed
task remains four-choice matching of an 8-frame, two-second wrist-video window
to its timestamp-matched 6-channel
IMU window, with three same-recording distractors at least three seconds away.

The two selected source sessions are `g154` (train) and `g119` (development),
both in category `cleaning_kitchen`. The same two fixed files were downloaded
twice to measure transfer variability; this does not add unique data or change
the split. The second acquisition is a timing repeat, not an extra sample.

## Integrity and conversion

Plan manifest:
`manifests/pretrained_reuse/egoviz-unseen-session-temporal-plan-20261029.json`

- Plan SHA-256: `8f19a087ab0d63031ce28045def2767acfe03290951da0f4b2abfc327428672c`
- Acquisition status: train/development only; `sealed_audit_downloaded=false`.
- Both MCAP files matched the plan's byte lengths and SHA-256 receipts.
- The existing builder validated MCAP CRCs, strictly increasing stream
  timestamps, source hashes, and train/development group separation.
- Window corpus: 14 train and 21 development examples; 35 unique IMU windows
  shaped `[35, 6, 128]`; 480 train and 690 development video frames parsed;
  zero group overlap. Publisher action labels were not read.
- `imu-windows.npy` SHA-256:
  `4063d672cb31c0951125cdf2f0625ac484fbdfd08b54699f33f65e21c5ec2957`
- `train-samples.jsonl` SHA-256:
  `9616c49ffcb288eaa4e5c6d4bc723bffdb9ec79c92d45a4004f8a0f49a49b9`
- `development-samples.jsonl` SHA-256:
  `d83f43c9bfc476025a630afb286b4341ecd0c4dfbe5e4ca2e27d3cc82333dad2`

## Frozen feature extraction

The same 35 samples passed through the already-pinned encoders in CPU-only
processes. Their local checkpoint SHA-256 values and source commits matched the
existing extraction scripts:

| Encoder | Checkpoint SHA-256 | Source commit | Device | Output | Elapsed | Feature SHA-256 |
|---|---|---|---|---|---:|---|
| VideoPrism-B | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` | JAX CPU | `[35, 8, 768]` | 95.09 s | `a77a315fe6f086a1db5d36cf0f65fe9ab48f9114d21f0f0a433350d4b3a284dc` |
| MOMENT-1-small | `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877` | `38f7310ad594100747ca2a8357e9c7ca7d323e0e` | PyTorch CPU | `[35, 6, 16, 512]` | 1.79 s | `f5239aa4857a7d972a905c76ef5fe7446da0ebaccff35b5500448e68dfc22970` |

Both arrays contain finite values. Video features are keyed by the 35 unique
sample IDs; sensor features cover dense window indices 0–34, and every
four-option example refers to four in-range candidate windows with a valid
target index. Train/development groups remain disjoint. The VideoPrism process
reported JAX 0.6.2 and `TFRT_CPU_0`; MOMENT reported PyTorch 2.6.0+cu124 and
Transformers 5.6.2 while executing on CPU. The expected upstream warning that
MOMENT's reconstruction head is pretrained was emitted. Feature extraction
does not establish task quality or sensor suitability.

External raw artifacts are under
`C:\CodexArtifacts\egoviz-unseen-group-temporal-smoke-20261030-attempt-01\`
and the timing repeat under
`C:\CodexArtifacts\egoviz-unseen-group-download-timing-20261030-attempt-01\`.
They are not checked into Git.

## Transfer observations and estimate

All four transfers have matching hashes across their repeats:

| Split | Bytes | First transfer (s) | Repeat (s) | SHA-256 prefix |
|---|---:|---:|---:|---|
| train | 256,307,528 | 20.98 | 12.86 | `eafeae93847b` |
| development | 260,539,513 | 17.43 | 12.73 | `e79bed9ccb63` |

Mean: **16.00 seconds per file**, standard deviation 3.44 seconds; pooled
throughput: approximately **15.4 MiB/s**. The plan contains 24 train and 20
development files totaling 9,733,562,633 bytes. At the measured pooled rate,
raw transfer is approximately 10 minutes. Allow **10–12 minutes** for transfer,
per-file requests, and hash/receipt overhead; this is an estimate from only two
unique files and two repeated transfers, not a service guarantee.

At the end of the transfer stage C: had 4.27 GiB free. After CPU feature
extraction, C: reported 2.25 GiB free; the system pagefile showed 3.48 GiB
current use and a 4.11 GiB peak. The timing correlation does not prove that
pagefile growth caused the disk change. RTX 3080 remained occupied by the
separate Orbitune process at 100% utilization and 419 MiB free; no GPU work was
started. Bulk media acquisition remains gated on safe disk space. The prior
CPU VideoPrism run took about 530 seconds for 216 windows; this smoke took 95
seconds for 35 samples, including decode and JAX initialization.

## Next gate

After disk capacity is confirmed, acquire train/development media with the
existing hash-checked downloader in fixed-plan order. Its split and offset
options allow selecting one plan row at a time without modifying the frozen
manifest. Preserve a minimum 3 GB free-space reserve. Keep sealed media
unopened. Reconfirm active GPU resources before any
GPU task; CPU-only frozen feature extraction is preferred if the pinned
VideoPrism and MOMENT weights plus their exact local runtimes are available.
Only then run the preregistered fixed scorer and controls. Do not use the
35-example smoke to select a model or claim generalization.

## Per-session aggregation replay (2026-10-30)

To support one-at-a-time media acquisition and feature extraction without
retaining the complete raw corpus, a CPU-only aggregator was added at
`scripts/aggregate_egoviz_unseen_group_features.py`. It accepts one package per
planned train/development session, verifies the frozen plan and split
identities, sample/corpus and feature hashes, pinned encoder environments,
feature shapes/finiteness, target-to-timestamp correspondence, and rebases
session-local sensor-window indices into combined train/development artifacts.
It rejects missing/duplicate plan rows and any sealed-audit session directory;
it does not fit a readout or evaluate quality.

The integration replay used only the existing `g154` train and `g119`
development packages (14 train / 21 development examples, 35 windows total).
The generated report is
`C:\CodexArtifacts\egoviz-unseen-group-temporal-smoke-20261030-attempt-01\aggregate-replay-02\aggregate-report.json`,
SHA-256 `c06c4e494f4e1f489641a6f3c4ba52704e7293107c20380d5c4d995a6aaf35f7`.
The report records two sessions, zero train/development group overlap, and no
sealed data. Four new synthetic tests cover index rebasing, missing-session
rejection, sealed-session rejection, and target/timestamp mismatch rejection.
This validates the combiner on two sessions only; it does not establish that
the full 44-session set has been acquired or that the task is learnable.

The aggregator now also supports `--validate-chunk-id` to validate one
train/development package before the full aggregate exists. For example:

```powershell
python -m scripts.aggregate_egoviz_unseen_group_features `
  --plan <frozen-plan.json> --runs-root <session-packages> `
  --validate-chunk-id <planned-chunk-id>
```

This reads only that planned session package and fails closed for unknown or
sealed IDs. The smoke replay packages were made against a two-session derived
plan, not the 44-session frozen plan; passing the derived plan validates the
replay packages, while supplying the full frozen plan correctly rejects their
different acquisition-plan hash. Future per-session acquisition must use the
frozen plan directly so its receipt can be validated against that plan.

At the 2026-10-30 00:01 JST resource check, C: had 2.23 GiB free and the RTX
3080 Laptop had 419 MiB free. No further raw media acquisition or feature
extraction was started. The full per-session plan remains gated on materially
more free disk space and a fresh resource check.
