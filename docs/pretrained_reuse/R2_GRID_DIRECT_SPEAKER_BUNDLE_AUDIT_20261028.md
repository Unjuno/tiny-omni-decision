# R2 GRID direct speaker bundle audit (2026-10-28)

Status: **one-speaker source audit confirms the official s13 bundles agree, but
the frozen validation manifest's s13 utterance IDs do not.** The existing GRID
pilot remains blocked and must not be scored or repaired by speaker-folder
remapping.

## Evidence

The official Sheffield corpus page says that audio, video and word alignments
are distributed separately for each talker. For talker 13, the direct source
URLs were:

- Audio: <https://spandh.dcs.shef.ac.uk/gridcorpus/s13/audio/s13.tar>
- Word alignments: <https://spandh.dcs.shef.ac.uk/gridcorpus/s13/align/s13.tar>
- Video: <https://zenodo.org/api/records/3625687/files/s13.zip/content>

The individual audio archive is 86,077,440 bytes (SHA-256
`7f04d74aaa7ee5f40512c822c162efa4d20d2b0312ac4ed3cde79f616aee56d3`) and has
1,000 WAV filename codes. The individual alignment archive is 1,034,240 bytes
(SHA-256 `964d9bb55c3257d9d1a308065219ec1ecec74f7354ab2051fc91563cc4add851`)
and has 1,000 alignment codes. The Zenodo s13 video ZIP is 407,686,627 bytes;
the provider lists MD5 `17e34a4628d75c789feecfdc38a45718`. Only its final 262,144
bytes were fetched (HTTP 206, range `407424483-407686626`), enough to parse the
complete ZIP central directory; no video payload was downloaded or read.

The three direct official s13 bundles share all 1,000 six-character utterance
IDs: audio/alignment 1,000; audio/video 1,000; alignment/video 1,000. In
contrast, none of the 1,000 rows labelled s13 in the frozen R2 validation JSON
match the direct official s13 audio, alignment, or video IDs. This is stronger
than the earlier `audio_25k.zip` same-name-folder audit: it shows the current
frozen validation manifest itself is not aligned to the official s13 bundles.
It does not establish why that manifest was misassigned, nor establish a
global speaker mapping for other talkers.

The reproducible audit is
`scripts/audit_grid_direct_speaker_codes.py`; its external receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-official-speaker-audio-20261028-attempt-01/s13-direct-speaker-audit.json`.
Receipt SHA-256: `a89ba3921564f202c16d71d21e89088eb3101aaf8f6271f0e2a96c45f1751e13`;
video metadata-tail SHA-256:
`a0e9c6b833a971f67819dc680032ad4334f38c3253a0d9d5e6616f7e118a5068`.
The command exited 2 intentionally with
`OFFICIAL_BUNDLE_MATCH_FROZEN_SPLIT_MISMATCH`. The receipt records input hashes,
counts, overlaps, and that no media payload or identity inference was used.

## Consequence

Do not use the old frozen GRID selection for AV inference. Do not remap its
speaker labels from filename overlap. Keep its selection and hashes unchanged
as a failed integrity receipt. If GRID remains useful, construct a new
speaker-grouped manifest from the official per-talker alignments, verify it
against the matching official audio/video code inventories, and freeze the new
sample before reading media or predictions. Repeat this source check for the
entire proposed split; this one-speaker audit does not clear the other seven
validation speakers or training partition.

## Reproduction

```powershell
python scripts/audit_grid_direct_speaker_codes.py `
  --audio-tar C:\CodexArtifacts\pretrained-reuse-r2\grid-official-speaker-audio-20261028-attempt-01\s13.tar.part `
  --alignment-tar C:\CodexArtifacts\pretrained-reuse-r2\grid-official-speaker-audio-20261028-attempt-01\s13-align.tar `
  --video-metadata-tail C:\CodexArtifacts\pretrained-reuse-r2\grid-official-speaker-audio-20261028-attempt-01\s13-video-tail.bin `
  --video-range-start 407424483 `
  --video-total-size 407686627 `
  --frozen-validation C:\CodexArtifacts\pretrained-reuse-r2\grid-development-split-20261028-attempt-02\validation.json `
  --speaker s13 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\grid-official-speaker-audio-20261028-attempt-01\s13-direct-speaker-audit.json
```

The nonzero exit is the expected integrity-gate result for this frozen
manifest. Tests for ZIP range parsing and the distinct source-match versus
frozen-split-mismatch states are in `tests/test_grid_direct_speaker_codes.py`.
