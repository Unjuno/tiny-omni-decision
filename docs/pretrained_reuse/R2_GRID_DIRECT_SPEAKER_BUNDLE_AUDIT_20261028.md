# R2 GRID direct speaker bundle audit (2026-10-28)

Status: **the direct official bundles agree for all eight validation speakers,
but four speaker groups in the frozen validation manifest have zero filename
overlap with their same-label official bundles.** The existing GRID pilot
remains blocked and must not be scored or repaired by speaker-folder remapping.

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
complete ZIP central directory; no s13 video payload was downloaded.

For the eight-speaker comparison, all eight complete per-speaker audio tar
archives were downloaded to the external artifact directory. Their SHA-256
checks read the complete archive bytes, and tar directory headers were parsed;
the WAV members were not extracted, decoded, listened to, or passed to a model.
For seven speakers, only 262,144-byte video ZIP tail ranges were fetched. The
s30 ZIP had already been fully downloaded and matched its publisher MD5, but
no video member was decoded for this source audit.

This audit was expanded to all eight speakers in the frozen R2 validation
split (`s13`, `s18`, `s23`, `s24`, `s25`, `s27`, `s30`, `s4`). For every speaker,
the direct audio, alignment and video inventories contain 1,000 codes and
intersect pairwise at 1,000. The frozen manifest's corresponding code rows,
however, match all codes for `s24`, `s25`, `s30`, and `s4` (999/999 for `s4`),
and match **0/1,000** for `s13`, `s18`, `s23`, and `s27`. Those same four labels
were affected in the combined-audio-archive folder audit. Direct source files
are therefore coherent; the previously frozen validation manifest has
incorrect code-to-speaker assignments for four of its eight groups. This does
not establish why those rows were misassigned and does not clear the train
partition.

The four mismatched groups account for half the speakers in the fixed pilot.
Keep the old pilot selection hash
`b33e4e7e2cdb77ffdfa58da371fd712dd57c63341651cfb416505adf57ccbf0d` unchanged
as a failed integrity receipt; it has not been evaluated. Do not repair it by
mapping code overlap to another speaker folder.

## Replacement pre-media pilot

After the eight-source comparison, a new development sample was generated
from the exact intersection of each speaker's direct audio, alignment and
video filename inventories. It retains the same eight speaker labels and the
same two-slot quartet construction rule, but gets new code-to-speaker records
and a separate deterministic selection namespace. Every speaker had 1,000
verified codes, 1,000 eligible two-slot quartets, and zero anchors without
three eligible neighbors. The replacement contains 64 quartets / 256 balanced
query roles, with 255 distinct audio codes and 255 distinct video codes.

The current selection is frozen outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/grid-av-direct-official-pilot-20261028-attempt-03/`.
`selected.jsonl` SHA-256:
`e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281`;
`summary.json` SHA-256:
`cf572ae2e71ccfc1acd939d829b5880d897175422a964e8fe0c5d064d6022412`.
The manifest was built by
`scripts/build_grid_direct_speaker_pilot.py`. The source-selection rule was
fixed before the source check. The builder reads tar/ZIP filenames and fixed
validation identities and hashes the complete audio tar archives; it does not
extract or decode media content. No predictions or sealed audit were read.
The two earlier isolated builder runs at attempts 01 and 02 produced the same
selected-list hash; attempt 03 is the authoritative receipt with corrected
archive-read accounting. This is a pre-inference development selection, not a
clean independent audit: cross-corpus speaker identity remains unknown, and
the separate train partition still needs a source audit before any GRID
training. The corrected receipt distinguishes hashing the full audio archives
from decoding the media.

The reproducible audit is
`scripts/audit_grid_direct_speaker_codes.py`; all eight per-speaker receipts are
embedded in the current external builder summary above. Four speaker receipts
report `OFFICIAL_BUNDLE_MATCH_FROZEN_SPLIT_MISMATCH` and four report
`PASS_DIRECT_SOURCE_BUNDLE_AND_FROZEN_SPLIT`. The receipts record each direct
archive hash, the video central-directory range, counts and overlaps. No media was decoded or
semantically inspected, and no identity inference was made. For seven video
archives only the final
262,144-byte ZIP metadata range was fetched. The s30 video ZIP was already
fully downloaded and passed the Zenodo publisher MD5; no other video archive
was fully downloaded.

## Consequence

Do not use the old frozen GRID selection for AV inference. Do not remap its
speaker labels from filename overlap. Keep its selection and hashes unchanged
as a failed integrity receipt. The separate replacement sample above has been
frozen from verified per-talker code inventories and can proceed to fixed-model
validation inference with controls. Repeat source validation for the full
24-speaker training partition before using those records. The current audit
clears only the eight direct validation speaker bundles; it does not clear the
train partition, cross-corpus identity independence, rights for product use,
or AV quality.

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
