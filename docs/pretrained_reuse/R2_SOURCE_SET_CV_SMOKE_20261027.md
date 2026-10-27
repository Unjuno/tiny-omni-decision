# Three-arm source set-decision wiring smoke

Using pinned frozen train features and the fixed group assignment, a separate
CPU process initialized each of the audio-only, video-only and joint models
with seed 17, performed one optimizer update on eight train rows, and ran the
set decoder and metrics on eight held-out video-group rows. All losses and
computed gradients were finite. Each readout has 344,834 parameters. The
observed losses were 1.11748, 1.11839 and 1.11739 respectively. This is a
wiring smoke, not a seven-epoch comparison, a reproducible quality metric or a
product assessment. No checkpoint, prediction file or audit was produced.
CUDA was not initialized.

Reproduce: `python -m scripts.smoke_source_set_cv` from the project root with
the pinned local features available. All modes check the source input hashes.

Before the full run, an additional limitation was recorded in the protocol:
with visual features set to zero, the audio-only candidate logits are identical.
The initial fallback selected the first candidate on a tie, creating an
ordering bias. Before any full run, the decoder was corrected to select all
tied maxima and the protocol was updated. The audio-only arm consequently
selects every candidate when it decides a source is visible; it cannot
establish visual source identification. Tests check tied logits and fallback.
This does not alter earlier results. The fixed comparison remains unrun due
to disk capacity.
