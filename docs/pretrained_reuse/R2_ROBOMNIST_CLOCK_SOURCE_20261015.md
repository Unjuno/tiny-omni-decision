# R2 — RoboMNIST public collection-clock evidence (2026-10-15)

**Read-only evidence inspection, not common-clock admission.** Six immutable
source text files were fetched after the fixed raw-input smoke. No source code
was executed, and no further media or positionGT was opened. A complete pinned
Git tree and exact Git blob SHA-1/byte-count/SHA-256 checks bind each file;
[the small receipt](R2_ROBOMNIST_CLOCK_SOURCE_20261015.json) records all URLs and
hashes. Local originals are in
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-clock-source-inspection-01`.
The revision is `70f5e6393acf6128a7ceb7afacf4463060626ca3`.

## What the implementation proves, and what it does not

| Source evidence | Observed implementation | Limitation for the released sample |
|---|---|---|
| [heartbeat_node.py](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/monitor/src/pack_wifi/pack_wifi/heartbeat_node.py) | A timer sends packet counters, then a terminal message; waits for configured collectors' completion. | Synchronization intent is visible; deployed configuration/callback delays are not recorded in released media. |
| [camera_node.py](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/monitor/src/pack_camera/pack_camera/camera_node.py) | Reads each camera sequentially at heartbeat; writes collected frames with fixed configured FPS. | No per-frame capture wall clock is written here. MP4 PTS is a playback clock, not proof of simultaneous exposure. |
| [csi_node.py](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/monitor/src/pack_wifi/pack_wifi/csi_node.py) | UDP reception uses time.time_ns(); heartbeat copies packets into per-IP collectors; writes a pickle .dat. | Raw receipt time differs from the released JSON's relative, identical three-module timeline. The transformation is not established by these files. |
| [params.yaml](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/monitor/src/pack_util/param/params.yaml) | Example frequency30Hz,packet_num360; collector list camera_node/csi_node; dimensions2560×720. | The actual pilot contains450 video/CSI points; this example cannot be asserted as its deployment config. |
| [README](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/README.md) and [camera setup.py](https://github.com/SiamiLab/RoboMNIST/blob/70f5e6393acf6128a7ceb7afacf4463060626ca3/monitor/src/pack_camera/setup.py) | README describes microphone collection and an audio broadcast command. Camera package exposes only camera/camera_show entry points. | The complete pinned tree contains neither mic_node nor audio_broadcast implementation. WAV start/stop/chunk handling is unknown. |

The pinned example CSI-reading notebook only reads the JSON and converts
time_stamp_ns to seconds; it does not establish the raw-to-release export or
the source of a shared relative clock. No raw pickle was loaded. Source package
metadata contains an unresolved code-license placeholder; dataset CC BY 4.0
metadata is not a license grant for arbitrary source-code reuse. This inspection
reads format facts without adopting upstream implementation code.

Heartbeat code also invokes synthesized spoken seconds during collection.
Whether this was active in the released recordings is **unknown**. It should
be checked as a possible shared acoustic/time cue before a quality experiment;
its presence or absence cannot be inferred from the README or audio RMS.

## Decision for the next stage

Retain `UNVERIFIED_COMMON_CLOCK` for the measured pilot. Do not use an assumed
samplewise audio/video/CSI alignment, interpolate a synthetic common clock, or
reinterpret450 equal counts as hardware synchronization. The public heartbeat
implementation supports a hypothesis of coordinated collection, not measured
latency bounds or proof that it produced this release.

An episode-level robot activity task could use same-repetition inputs without
claiming subframe correspondence. Before fitting it still needs an explicit
task/option contract, unchanged source grouping, verified session/overlap gates,
label-free tensors and unimodal/joint controls on the same examples. Fine temporal
localization additionally requires capture/export clock evidence. Neither path
is admitted by this inspection, and no next training run was started.

This source supplies a narrow robot-domain candidate; even strong results here
would not demonstrate general text/image/audio/video quality or the95–98% Omni
target. The existing failed, incomplete and overfitting results remain intact.
