# Perception AV event-source coverage scale results (2026-10-21)

## Result and scope

**Execution: PASS. Quality interpretation: UNCERTAIN / task-specific.** Four
CPU readout fits completed the prespecified 2,048 optimizer updates. The
comparison asks whether adding 233 previously unused fit videos (550 events)
helps the existing conditional audio-event to visible-object-source task, at a
matched update budget. It does not measure general AV reasoning, four-modality
quality, or the product target. The official TRAIN annotations had already
been parsed during feasibility work, and the development split is not blind.
The previously scored 200-video holdout was not opened or used.

The joint development Accuracy point estimate changed by -0.7 percentage
points, with a video-cluster 95% interval spanning zero. The parameter-matched
video-only Accuracy changed by +4.8 points, but its interval also spans zero;
its Brier and ECE worsened. These results do not support a reliable quality
gain from the extra fit coverage. They also do not rule one out given the small
development set and single seed.

## Provenance and frozen data

- Branch: `codex/r2-perception-av-scale-v1`.
- Source commit for the PTS audit, feature extraction, and fits:
  `334d2f5f5d0d555db4cff69c2d7fe25b1a0d4431`.
- The plan's source commit is `77c2949c6d454ad81ba335f2a0b2a1601bbb8297`;
  an earlier draft's incorrect commit transcription was superseded without
  changing this pinned plan.
- Frozen plan:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-plan-03\plan.json`,
  SHA-256 `43dda454adf45352dee281680f51698ecf38d60731d33311a6ded139de167461`.
- Acquired media: 388 videos, 5,692,348,837 bytes; acquisition completed with
  `PASS_ACQUIRED_UNVERIFIED_DECODE`. No test partition was opened. Acquisition
  and retries incurred no cloud or paid-compute cost.
- Full PTS/media integrity audit:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-pts-01\audit.json`,
  SHA-256 `785c6c75648f9cd1a85cddbed868bf35235a803b9f1439ceb8a651c9081f53b5`.
  All 908 events had complete video support and nonzero audio overlap; none
  were right-censored. Across the previous and new media, 2,179 unique media
  hashes were checked.
- Frozen feature extraction completed all 388 videos / 908 events in 867.59
  seconds on the local RTX 3080, through the HotUGPU guard. Peak child CUDA
  allocation was 340,768,256 bytes; this excludes the wrapper's separate VRAM
  reservation. No process was killed or restarted. The wrapper waited for the
  GPU reservation before launching extraction.
- Feature result SHA-256:
  `7907a617c97037fcabb001d46be70a2dc48c5487156c85d46b1463a3d8d0dd43`;
  feature manifest SHA-256:
  `7b0c7b19d074744fe4447b160335d1e9a6d7587243516bb84f3fdcf31ce2a5a1`.
- The fit comparison used old-only manifest SHA-256
  `996cda913028485cde188cd1f8822868986edca80b4d64087a28de452f73c837`
  (525 fit events / 240 videos) and expanded manifest SHA-256
  `54dd89b9a72ec80e679fb9fa653c79ecfd27d6a6a4aaa8a42120c767b2ca958b`
  (1,075 fit events / 473 videos). Both arms have identical selection
  (121 events / 58 videos), calibration (91 / 39), and development validation
  (146 / 58) identities. Video groups remain within a single split.

## Fixed readout comparison

All runs used seed 17, CPU, AdamW, learning rate 0.001, hidden size 128 for
joint and 339 for video-only, evaluation every 512 updates, 2,048 maximum
updates, min-delta 0.0001, and patience 20. Selection was minimum video-disjoint
selection NLL. Best and final weights were kept separately, and best-checkpoint
reload checks passed. There was no patience stop. Joint has 246,658 trainable
parameters; video-only has 246,115. All conditions used the same 58-video
selection and 58-video development validation groups.

### Best checkpoints and development validation

| Readout | Fit coverage | Best step | Selection Accuracy / NLL | Dev Accuracy | Dev NLL | Dev Brier | Dev ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Joint | old-only | 1,536 | 0.5372 / 1.7251 | 0.3356 | 2.0732 | 0.8073 | 0.1769 |
| Joint | expanded | 2,048 | 0.5124 / 1.6558 | 0.3288 | 2.0322 | 0.8100 | 0.1983 |
| Video-only | old-only | 512 | 0.3388 / 1.9491 | 0.2466 | 2.1593 | 0.8399 | 0.1208 |
| Video-only | expanded | 2,048 | 0.3967 / 1.8292 | 0.2945 | 2.1534 | 0.8609 | 0.1977 |

Development validation contains 146 events from 58 videos. Paired bootstrap
resampled videos, 10,000 draws, seed 17, and reports expanded minus old-only:

| Readout | Accuracy delta (95% video-bootstrap interval) | NLL delta (95% video-bootstrap interval) |
|---|---:|---:|
| Joint | -0.00685 `[-0.09741, 0.07759]` | -0.04100 `[-0.26956, 0.17778]` |
| Video-only | +0.04795 `[-0.03185, 0.12500]` | -0.00595 `[-0.16196, 0.15035]` |

Both intervals include zero. Joint Accuracy is effectively unchanged at this
sample size; its NLL point estimate improves while Brier/ECE worsen. Video-only
Accuracy rises as a point estimate, while NLL is nearly unchanged and Brier/ECE
worsen. Do not treat any of these as a confirmed gain.

### Learning curves

Train CE is the mean over optimizer updates since the previous validation; it
is rolling training behavior, not a fixed-example generalization gap.

| Readout / coverage | Step 512 | Step 1,024 | Step 1,536 | Step 2,048 |
|---|---|---|---|---|
| Joint / old-only | 1.9860 / 1.9151 / .3388 | 1.4837 / 1.9824 / .3967 | 1.0464 / 1.7251 / .5372 | .5918 / 2.1446 / .4711 |
| Joint / expanded | 1.9441 / 1.8412 / .4215 | 1.7576 / 1.8248 / .4215 | 1.4949 / 1.7313 / .4132 | 1.4781 / 1.6558 / .5124 |
| Video-only / old-only | 1.9885 / 1.9491 / .3388 | 1.6660 / 2.0623 / .3471 | 1.5738 / 1.9883 / .3719 | 1.4857 / 2.0221 / .3719 |
| Video-only / expanded | 1.9709 / 1.8805 / .3967 | 1.8270 / 1.8997 / .3967 | 1.6847 / 1.8338 / .4215 | 1.7317 / 1.8292 / .3967 |

Each cell is `rolling train CE / selection NLL / selection Accuracy`. The
old-only joint curve shows a clear late overfit signal: train CE continues to
fall while selection NLL worsens at step 2,048. Expanded joint selection NLL
continues to improve through the final checkpoint. The video-only curves are
mixed and do not show a stable monotonic selection improvement.

## Reproduction and artifacts

The local Python installation has a stale editable package path into an
unrelated worktree. Commands therefore set `PYTHONPATH=src` and did not modify
or upgrade the environment. The four fit invocations used the same runner and
arguments, substituting the feature manifest, condition, and fresh output
directory:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m scripts.train_av_event_source --features <old-or-expanded-features.jsonl> --condition <joint-or-video_only> --output-dir <fresh-run-directory> --seed 17 --epochs 100 --patience 20 --learning-rate 0.001 --min-delta 0.0001 --hidden-size <128-or-339> --max-updates 2048 --evaluation-interval 512
python -m scripts.analyze_perception_av_event_scale --old-features <old-only-features.jsonl> --expanded-features <expanded-features.jsonl> --old-joint <old-joint> --expanded-joint <expanded-joint> --old-video-only <old-video-only> --expanded-video-only <expanded-video-only> --output <fresh-analysis-directory>/result.json --bootstrap-draws 10000
```

The frozen extractor used the HotUGPU wrapper with a 3,200 MiB reservation,
14,000 MiB free-memory admission threshold, 8,000 MiB minimum child memory,
image batch 16, and CUDA device. It used DINOv2-small revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056` and CLAP revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`; both encoders were frozen. Runtime
was Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, and PyAV 18.1.0.

All external run artifacts are preserved under
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-runs-01`.
The analysis result is
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-analysis-02\result.json`,
SHA-256 `4de1f6288f2680695226dc2e54822b956f0a668ea7ce4208ce1189e4c14b4e05`.
It includes prediction, result, manifest and bootstrap hashes. Media, features,
and checkpoints remain outside Git. Cost: local CPU and one gated local GPU
feature-extraction run; cloud spend `$0`.

## Limitations and next steps

- Single seed and one narrow event-source task; not a general AV or
  multimodal-decision benchmark.
- Development validation is small, previously parsed, and not blind. The
  intervals quantify video-cluster resampling uncertainty, not data-source or
  annotation-exposure bias.
- Public annotations do not provide participant identity, so participant-level
  disjointness is unknown.
- A historical TLS EOF was recorded in the acquisition receipt after a
  failed byte-range attempt; the identical pinned range was retried, hashes and
  lengths were validated, and the completed receipt passed. A zero-byte
  unregistered partial file was excluded and preserved.
- The analysis compares fit coverage under the same update budget; it does not
  isolate distribution shift or establish that both modalities are necessary.
- No sealed audit, test partition, model promotion, ternary quantization, or
  additional training run was performed.

Retain this as narrow development evidence. If continuing this task, first
improve the quantity and diversity of video-disjoint development evidence and
document an approved participant/source grouping strategy; do not select a
product model from these point estimates.
