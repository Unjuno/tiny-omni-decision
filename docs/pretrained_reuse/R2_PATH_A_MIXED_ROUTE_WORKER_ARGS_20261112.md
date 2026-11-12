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
inference occurred and Orbitune's process was left untouched. This is an
environmental runtime-smoke failure, so actual text inference with the mixed
configuration remains unverified.

External attempt directory:
`D:\CodexArtifacts\pretrained-reuse-r2\path-a-mixed-route-runtime-smoke-20261112-attempt-01`

- Result JSON SHA-256: `836879cc76ca5a11c6155ef188cdec1162ffef60a893c8f9bf99ee39eff9b99e`
- Worker log SHA-256: `115be7a8d990a215fe05362247dd48f242f1c0a3215c73f8e1cd00598c9440a0`
- Synthetic request SHA-256: `20735a76d700f54e41bb8bbb9a5783bea579d788e221d1be851bd38eddf103f7`
- Source commit: `c49127143fe483ecfb5b6c96af9e616dded72bc2`
