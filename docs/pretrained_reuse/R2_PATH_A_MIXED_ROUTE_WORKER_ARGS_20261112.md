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
