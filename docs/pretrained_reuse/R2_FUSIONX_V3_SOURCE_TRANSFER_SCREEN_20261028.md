# FusionX V3 source and transfer screen (2026-10-28)

Status: **REVIEW; transfer probe completed, not admitted for training or
quality evaluation.** No model, GPU, sealed audit, or training data split was
used.

## Source pin and rights

- Dataset: [TouchTronix FusionX Multimodal Sample Dataset V3](https://huggingface.co/datasets/touchtronix/FusionX-Multimodal-Sample-Data-V3)
- Immutable repository revision observed from the Hub API: `51392e3b05d8566b29ad8fe9f78478e8fef4971b`.
- The dataset card declares CC BY 4.0 and explicitly says commercial use is permitted with attribution to TouchTronix Robotics. The repository is public and not gated. This is the repository's stated license; this screen does not independently establish participant consent or publicity/privacy permissions.
- The card describes synchronized monochrome stereo, RGB, OAK-D IMU, and tactile glove data across 11 task folders. It lists 47,348 image rows, one `train` split, and a total payload of 15.9 GB.
- No independent validation/test split, actor/session grouping protocol, choice targets, or evidence that decisions require both vision and touch is supplied on the inspected card. The source currently cannot support a leakage-safe R2-M model-selection experiment.

## Bounded transfer measurement

The user asked for three small downloads and a wait-time estimate. To avoid
using the unlicensed MUVS candidate, the probe used three RGB files from the
explicitly CC BY 4.0 FusionX V3 release. Each request was pinned to the
immutable revision above and saved in a separate folder outside Git:

`C:\CodexArtifacts\pretrained-reuse-r2\fusionx-v3-download-probe-20261028-attempt-01\`

Each payload size and SHA-256 matched the Hub's LFS metadata. The timed span
includes the complete HTTP response body and local write, but excludes the
subsequent hash calculation.

| File | Bytes | Seconds | MiB/s | SHA-256 |
|---|---:|---:|---:|---|
| `strap1/rgb/000000.jpg` | 330,728 | 2.31 | 0.14 | `28f3c314f0b05ce8c5e205312aa313ff7a070f110b87cd4e23a39f9b2e6cd0de` |
| `strap1/rgb/000100.jpg` | 266,268 | 1.71 | 0.15 | `35a6cc96d89e75eb0be18857bbca822b17fcb3fba1d18406492d99b72f5a67a0` |
| `strap1/rgb/000200.jpg` | 269,952 | 1.51 | 0.17 | `c6704690f9e6d9320335308f72792fb9f8462850973bc3687f6e3b4bea379abd` |

- Total: 866,948 bytes (0.827 MiB) in 5.522 seconds.
- Arithmetic mean of the three rounded per-file rates: 0.153 MiB/s.
- Size-weighted rate: 0.150 MiB/s.
- At that weighted rate, a 15.9 GB decimal payload would take about 28.1 hours serially; if the displayed size is interpreted as 15.9 GiB, about 30.2 hours. This is a rough linear extrapolation from three sub-0.32 MiB files, not a reliable full-corpus ETA. CDN throughput, per-file request overhead, parallel downloads, retries, and local writes can change the result substantially.
- Available C: space at the check was about 0.81 GiB, so the 15.9 GB payload does not fit. No full or partial task tree was downloaded. The three files occupy less than 1 MiB.

One of the three inspected RGB frames shows a first-person indoor workstation,
computer displays, and a gloved hand. No face was visible in that frame, but
the workstation/display content makes the missing participant/privacy terms a
real review item. The frame was not used for model input, feature extraction,
training, or evaluation.

## Disposition

Keep FusionX V3 at `REVIEW`. Before any model-facing use, obtain authoritative
participant/privacy terms, inspect immutable task/episode/actor identities and
establish independent groups, define targets and a held-out validation split,
and preregister metadata-only, video-only, tactile-only, paired, and
cross-modal-mismatch controls. Do not begin a model run on the current
single-split release. The full-corpus transfer estimate is not a recommendation
to download it; current disk capacity is insufficient.

## Reproduction record

- Local source branch: `docs/pretrained-reuse-roadmap-20261010`.
- Source commit before this report: `d0bb69ef9721208936dc950016e5bfc8d30df7df`.
- Small-transfer receipt: `C:\CodexArtifacts\pretrained-reuse-r2\fusionx-v3-download-probe-20261028-attempt-01\transfer-probe.json`.
- The sample payload and receipt are outside Git. No legacy artifact, run,
  checkpoint, frozen corpus, worktree, or live process was modified.
