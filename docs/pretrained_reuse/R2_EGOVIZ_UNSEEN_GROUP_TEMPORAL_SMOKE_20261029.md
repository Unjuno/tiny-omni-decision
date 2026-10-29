# EgoViz unseen-session temporal smoke — 2026-10-29

## Scope and outcome

This is a raw-media and synchronized-window pipeline smoke for the frozen
unseen-session plan. It is **not** a model-quality evaluation: no encoder
features were extracted for this smoke, no model was fit, and no sealed-audit
media or labels were accessed. The fixed task remains four-choice matching of
an 8-frame, two-second wrist-video window to its timestamp-matched 6-channel
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

At the end of this smoke, C: had 4.27 GiB free and the RTX 3080 Laptop had
419 MiB free VRAM while the separate Orbitune training process remained active.
No GPU work was started. Bulk media acquisition and feature extraction remain
gated on safe disk space and a verified local checkpoint/runtime path. The
prior CPU VideoPrism run took about 530 seconds for 216 windows, but that
historical timing is not used as a promise for this corpus.

## Next gate

After disk capacity is confirmed, acquire train/development media with the
existing hash-checked downloader while preserving a minimum 3 GB free-space
reserve. Keep sealed media unopened. Reconfirm active GPU resources before any
GPU task; CPU-only frozen feature extraction is preferred if the pinned
VideoPrism and MOMENT weights plus their exact local runtimes are available.
Only then run the preregistered fixed scorer and controls. Do not use the
35-example smoke to select a model or claim generalization.
