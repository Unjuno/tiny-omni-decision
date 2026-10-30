# FSD50K selective training-media smoke

Acquired eight official-train clips without downloading the18GB archive.
Selection was fixed before fetching: ordered event list, CC0/CC BY only,
SHA256(`17:<clip_id>`) ordering, one clip per event, no repeated clip or uploader.
No selection on model scores, duration or decoded content; no replacements.
This covers transport/runtime feasibility, not a balanced training corpus.

| Positive event | Clip | Decoded seconds |
| --- | --- | ---: |
| Bell | 29792 | 2.202 |
| Clapping | 432910 | 1.126 |
| Telephone | 38229 | 5.696 |
| Walk_and_footsteps | 233384 | 8.689 |
| Musical_instrument | 239746 | 6.250 |
| Knock | 209222 | 1.943 |
| Door | 319041 | 3.007 |
| Water_tap_and_faucet | 273222 | 20.376 |

All eight are mono44.1kHz and decode to finite arrays. Decoded hashes are distinct
within these eight; cross-source exact/near duplication remains unverified.
Clip-level positive annotations must not be blindly assigned to every5s crop,
especially the20.4s faucet clip. Absence remains unknown, not negative.

First attempt failed before audio retrieval because Python zipfile rejects
multi-disk archives. Preserved its selection, error and executed source in
`C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-smoke-01/`.
The repaired path parses the original central directory in a directory-only
envelope, retaining each member's original disk and local offset. A unit test
verifies those fields survive. Fetching checks exact HTTP ranges, local filename,
compression method, expanded size and ZIP CRC. Whole-archive MD5 is **not**
verified by partial retrieval; each acquired file has a saved SHA256.

Second attempt completed in
`C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-smoke-02/`:
6,888,181 transferred bytes including directory/header data. Selection matches
the first attempt exactly. All metadata hashes were checked before selection.
TLS validation remains enabled. No evaluation member, GPU, new dependency,
training run or old artifact was used or changed. PyAV was used for the separate
decode check because soundfile is not installed; no environment upgrade.

Command: `python -m scripts.fetch_fsd50k_training_smoke`.
Ruff passes; full CPU suite744passed/1skipped. Companion JSON retains selection,
attribution/license, CRC/SHA, central-directory hash and decoded measurements.
Next: verify cross-source separation and a masked-positive training protocol
before expanding acquisition. This smoke is not a quality result or product gate.
