# CAV-MAE Sync real-input mechanics

Eight existing Perception Test **training** recordings passed pretrained CPU
feature extraction, with finite outputs and bitwise repeat equality. No labels,
validation selection, fitting, or sealed data were used. This is not accuracy
evidence. Measured extraction/repeat loop: 11.83 seconds, excluding model load.

Command: `python -m scripts.smoke_cav_sync_real_input`.
Input manifest SHA-256:
`e070ab02b518046703f1d76d37119b37186ab883cfc8a20bdf7eb06a84c784fd`.
Each existing window's hash is checked before loading. Checkpoint strict load and
SHA verification follow the previous CPU audit; weights and all legacy inputs
remain unchanged.

## Temporal adaptation

These approved recordings exceed the upstream ten-second input convention.
Blindly keeping the first ten seconds would misalign the retained middle image.
Instead, take retained image index4/8 and use its actual PTS to center a 4.175s
audio segment, clamping to the already approved common interval. This produces
416 Kaldi frames at 10ms shift and 25ms window, with128 mel bins, no dither,
Hanning window and waveform mean removal. Normalize using upstream mean -5.081
and standard deviation4.4849. Image processing uses bicubic resize224,
center crop224 and ImageNet normalization. The stored frames were already
resized to288 square by the legacy window preparation; this is a smoke input
limitation, not a claim of exact original-video preprocessing parity.

No silent zero-input fallback, temporal augmentation or frame search is used.
This is a single-frame/local-audio check, not upstream16-frame retrieval or
temporal reasoning. Future task comparisons must fix preprocessing in advance.

Output: `C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-real-input-01/`.
`features.npz` contains8 audio and8 image CLS vectors, each768 dimensions; SHA:
`5a9464ae0ea0fa6464a733eff4de9df6dc36d06cea9a31747b82a1cc5417ad23`.
`result.json` records all IDs, timestamps, sample bounds, rate and hashes.

Research inspection only: product weight license remains REVIEW. Next useful
evidence is a fixed task comparison on the existing development protocol, not
more zero-input checks or a claim that correspondence implies Omni decisions.
