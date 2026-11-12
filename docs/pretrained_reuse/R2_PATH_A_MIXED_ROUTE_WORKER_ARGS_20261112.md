# Path A mixed-route worker argument fix — 2026-11-12

## Finding

When a Perception video readout was configured, `PathARuntimeDispatcher` also
added `--perception-video-readout` and `--text-model-dir` to the persistent
Torch worker command. Those options belong to the VideoMAE worker. The Torch
worker parser accepts `--assembly`, `--device`, its audio cache option, and the
Speech Commands readout option, but not the Perception/VideoMAE options. As a
result, starting a Text, Audio, or legacy video-score route in a runtime that
also had Perception video configured could exit during argument parsing.

## Change and check

The Perception-specific options remain on the VideoMAE command and are no
longer added to the Torch command. A CPU regression builds the Torch command
with Perception configured and asserts that its arguments match the Torch
worker contract. The existing VideoMAE command test continues to verify that
the Perception readout and assembly MiniLM path are passed to the correct
worker.

The regression was first run against the old code and failed because
`--perception-video-readout` was present in the Torch command. After the fix,
the focused dispatcher tests and Ruff pass.

This verifies command construction and route integration only. It does not
load checkpoints, access a corpus, run GPU inference, or provide task-quality
evidence.

## Pinned CPU runtime attempt

After the code fix, one label-free text request was attempted through the real
dispatcher with the Perception readout configured. The selected output folder
was new and on D:, CUDA was hidden with `CUDA_VISIBLE_DEVICES=-1`, and the
request used a synthetic state/question/options payload. The attempt stopped
before the worker parser or any checkpoint load: importing the pinned
`torch 2.6.0+cu124` environment failed while loading `cusolver64_11.dll` with
Windows `WinError 1455` (insufficient paging-file/commit capacity). No GPU
inference occurred and Orbitune's process was left untouched. This attempt
alone did not verify mixed-configuration text inference; the later attempt 06
below succeeded in an isolated CPU-only environment.

External attempt directory:
`D:\CodexArtifacts\pretrained-reuse-r2\path-a-mixed-route-runtime-smoke-20261112-attempt-01`

- Result JSON SHA-256: `836879cc76ca5a11c6155ef188cdec1162ffef60a893c8f9bf99ee39eff9b99e`
- Worker log SHA-256: `115be7a8d990a215fe05362247dd48f242f1c0a3215c73f8e1cd00598c9440a0`
- Synthetic request SHA-256: `20735a76d700f54e41bb8bbb9a5783bea579d788e221d1be851bd38eddf103f7`
- Source commit: `c49127143fe483ecfb5b6c96af9e616dded72bc2`

## Pinned CPU runtime attempt 06

The exact production JSONL dispatcher and unmodified Torch worker completed
one synthetic Text request while the Perception VideoMAE readout was configured.
This used a separate Python 3.11.9 CPU-only environment (`torch 2.6.0+cpu`,
`transformers 5.6.2`, `numpy 1.26.4`); CUDA was hidden. The Torch worker started
ready on CPU, reported 55,986,819 resident unique parameters, and did not
receive either Perception-only CLI flag. End-to-end dispatch took 8.7047 s.
The output was valid typed JSON and the three probabilities summed to 1 within
floating-point tolerance.

This is runtime evidence only. The request was synthetic, no dataset or media
was loaded, and no quality metric was evaluated. In fact, for the synthetic
arithmetic question whose correct option was 22, the model selected 18
(probability 0.5055); the smoke must not be interpreted as a correct decision.
No GPU was used and no Orbitune process was touched.

External attempt directory:
`D:\CodexArtifacts\pretrained-reuse-r2\path-a-mixed-route-runtime-smoke-20261112-attempt-06`

- Result JSON SHA-256: `d16358a62cd42f95967cee3b3a0f386beb582d7e84b7341f36a2b5b4ee40a787`
- Synthetic request SHA-256: `996c3652c9d22da414d821001f296224de5eaeadd10daa6381aec52bf6a5f7ec`
- Torch worker log SHA-256: `4ca0d533ed456b6de22534e44ef7e5aaa04d1660c004d7450b3da851cf874431`
- Environment freeze SHA-256: `5aa227d58caf2a1cc4a11c2c4b4303b6fd8f62f1a581b540310a3fb389e1f4d6`
- Assembly manifest SHA-256: `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`
- Perception readout SHA-256: `e9516be6d58ba82c3c4f111e59d6b724e856b3ac412bbc2c57b683d35deecb8c`
- Source commit: `b9716d7ea7b20fa8d0f9ec635249b018bd1068c5`
