# R2 — actual RoboMNIST native input mechanics (2026-10-15 JST)

**PASS for a fixed input-mechanics smoke and fresh-process array replay only.**
Training corpus admission, common-clock synchronization, task splits,
complementarity, OOD generalization and Decision quality remain unverified.
The full <=200M resident prequantization Omni / calibrated JSON / 95–98% goal is
not achieved. No model was loaded or fitted in this stage.

## Fixed source and execution

The [prospective protocol](R2_ROBOMNIST_INPUT_PROTOCOL_20261014.md) and producer
were committed before execution at
`cfa9e3dd121bfbd3a7f5a3cb5b12f0e51ebd7d53`.
RoboMNIST [version 1](https://doi.org/10.6084/m9.figshare.28179383.v1) has
CC BY 4.0 data metadata. The preceding complete-directory audit identified
2,135 repetition groups across all60 archives; this smoke uses exactly the
preselected first lexical group `Ar1_Vhigh_Ac0_Rp01_UNC062` from directory00.
This now-observed pilot must remain on the development/training side of any
future split. It cannot become a blind test example.

```powershell
python -m scripts.fetch_robomnist_input_smoke --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-smoke-01
python -m scripts.replay_robomnist_input --main C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-replay-02.json
```

Producer PID56168 started at2026-10-14T15:06:38.722225+00:00 and ended normally
at2026-10-14T15:09:44.745373+00:00 (2026-10-15 00:06:38–00:09:44 JST).
Status is `PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY`; exit code0.
Exactly seven complete source files were fetched: three cameras, three
microphones and one three-module CSI file. PositionGT was not fetched/opened.
All views belong to the same physical repetition. No retries, other groups,
annotation inspection, full archive download, sealed audit or GPU work occurred.

## Actual native formats and timing

| Input | Native data | Time extent / format |
|---|---|---|
| Camera1/2/3, each | 450 RGB frames,720×2560×3 | MPEG-4,30fps,time_base1/15360; PTS0–14.966666667s |
| Microphone1 | 658,432 samples | mono PCM16,44,100Hz;14.930430839s |
| Microphone2 | 659,456 samples | mono PCM16,44,100Hz;14.953650794s |
| Microphone3 | 660,480 samples | mono PCM16,44,100Hz;14.976870748s |
| CSI module1/2/3, each | 450×256 complex subcarriers +450 RSS values | timestamps0–14,966,323,249ns;14.966323249s |

All450 RGB frames per video were decoded and individually hashed in order.
The fixed three-frame retention rule selects indices0/224/449; a second native
decode verifies retained pixels and their PTS. Fresh-process replay verifies
all450 PTS per video. Stereo views are horizontally
concatenated in each720×2560 frame. No crop, resize or frame-rate conversion was
applied. WAV PCM uses the exact native16-bit conversion tofloat32; channels,
sample count and rate remain intact. CSI preserves all256 subcarriers asfloat64
real/imaginary values, native RSS andint64 timestamps, with finite-data,
shape, unique-module and strictly increasing timestamp checks. No evaluation of
source strings as code occurs.

In this actual sample the CSI timestamps are **relative, beginning at0**, not an
absolute epoch. All three modules have identical timestamp vectors; the median
interval is0.033329434s (min0.032217536,max0.034365279s).
Earlier conservative references to an absolute-clock feature risk do not imply
that an absolute epoch was observed. These timestamps are alignment metadata,
not a learned source/session cue.

The WAV files have no absolute start clock. Media PTS and similar durations
cannot certify their offset against CSI. Status remains
`UNVERIFIED_COMMON_CLOCK: media PTS/WAV do not certify CSI mapping`.
No temporal stretching, padding, resampling, shift fitting or assumed common
start was used. Equal video/CSI counts are insufficient synchronization evidence.

## Resources and checksums

| Measurement | Actual | Prospective bound |
|---|---:|---:|
| Producer time | 186.024231s | 900s |
| Response body bytes | 26,173,462 | 67,108,864 reserved transfer cap |
| Reserved bytes including length checks | 26,173,517 | 67,108,864 |
| Producer peak CPU working set | 191,918,080B | 2GiB |
| Producer output bytes | 51,220,288 | 384MiB |
| Fresh replay time | 12.058361s | 600s |
| Fresh replay peak CPU working set | 243,945,472B | 2GiB |

The1GiB disk reserve is specific to this bounded CPU data smoke. The interrupted
native GPU diagnostic retains its original8GiB guard and was not restarted.
No cloud cost or environment upgrade. Python3.11.9,NumPy2.4.6,PyAV18.1.0;
exact installed library/decoder/source files and FFmpeg versions are in the
effective config and replay pins. GPU observation was idle0MiB/0%.

HTTP206, Content-Range, exact response length and TLS remain checked. Every
request reserves bytes before transport. ZIP offsets, compressed/uncompressed
sizes, compression type and CRC match the parent directory; every full local
file has size/CRC32/SHA-256 checks. **The whole source archive's MD5 is not
verified**, because only the fixed seven members were fetched. This does not
establish cross-corpus content uniqueness.

## Replay, fixes and provenance

[Published replay02 JSON](R2_ROBOMNIST_INPUT_RESULTS_20261015.json) is an exact
copy of the separate-process raw replay. All27 saved arrays match the newly
decoded originals in shape, dtype and every byte. The consumer checks92 pins:
the81 producer/source/library inputs plus config/result/source binding, seven
raw files and the NPZ. The decoder/parser implementation is shared and pinned;
this is fresh-process execution, **not an independent codec implementation**.

Replay01 remains preserved as the initial successful raw replay. Self-review
found that mandatory new helper/decoder/library pins were not independently
required in its admission check. A regression test failed first; the consumer
now requires them and replay02 was run from scratch. Tests reject partial or
clock-PASS laundering, metadata/GT input leakage, inconsistent transfer bounds,
missing required pins and any array key/shape/dtype/byte mismatch.

The consumer was not committed when replay02 ran; its exact executed SHA-256 is
`b621a70d7b165b8e8cef0e8cecb93be1f6d4ca642065d1c9fa639feee3f0c83e`.
The producer source commit remainscfa9e3d, not the later publication commit.
Final current-file checks match all92 pins and this exact consumer SHA.

Three receiver midpoint frames were also inspected as lossless PNGs with exact
NPZ pixel readback, outside Git. They show horizontally paired views of robot
arms in a laboratory. No obvious task-class/action/speed text overlay was seen
in **these three frames only**. This does not certify all frames or the source
against visual leakage, identify which visible arm is the source label, or
justify a crop/encoder preprocessing choice. No positionGT was consulted.

| Artifact | SHA-256 |
|---|---|
| Producer result.json | `5936b74193b1fdf8f813d85a5eb2391c48123a3eeb671153a0aad89f1980227b` |
| Effective config.json | `7dbdb01d72ce3d55acdb88cc3c52fa0cf641d1b3dde2b029fb69e5c9664d1af8` |
| native-inputs.npz | `6e227e7873a5694165a467c8d8eb2713e563fee9916b048c4222a2ab0fd09f84` |
| Fresh replay02 JSON | `5cd718a66f53d843ce30d9f1ba3519aa534ca977319491d90798367096576d51` |
| Visual inspection receipt | `b546862e6ec5a0ab444c0e3a93ceb6f49fc1c6609d7aeb82a821b1f94a25c25e` |

Local roots are `C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-smoke-01`,
`robomnist-input-replay-01.json`, `robomnist-input-replay-02.json`, and
`robomnist-input-visual-inspection-01`. Raw media, NPZ and PNGs stay outside Git.
The source-format notebook is pinned as data at
`SiamiLab/RoboMNIST@70f5e6393acf6128a7ceb7afacf4463060626ca3`; no notebook was
executed and its code license is not assumed from the dataset's CC BY grant.

## Verification and next admission gates

Fresh verification: Ruff passes; CPU pytest **1,468 passed,1 skipped**;
NoTorch **1,054 passed,96 skipped**. Model and every candidate manifest, allfour
training/evaluation catalogs and the existing dataset-policy audit pass.
Read-only code review found no remaining Critical/Important blocker.

The subsequent [immutable collection-code inspection](R2_ROBOMNIST_CLOCK_SOURCE_20261015.md)
finds common heartbeat intent, but missing audio capture/export code and a
360-sample example that cannot establish this450-sample release's timing.
Define an honest temporal contract from further evidence; do not invent offsets
from these durations. Episode-level
tasks may need a different timing contract from fine temporal localization, but
neither is admitted merely by this smoke.

Before fitting: define supplied-option tasks/labels; remove source identifiers
from learned inputs; group every view by physical repetition and evaluate
session/source/normalized-content overlap; freeze train/selector/audit and
OOD rules; verify whole-corpus media and source rights; compare unimodal and
joint controls with identical cohorts. Actual public-source mechanics provide
a next route, not evidence that joint decisions are accurate.

Legacy v1/v2/QAT/native-memory artifacts remain protected. The8GiB disk-stopped
native memory job is still incomplete and unrestarted; no partial quality
scoring, product promotion, compression or main merge occurred.
