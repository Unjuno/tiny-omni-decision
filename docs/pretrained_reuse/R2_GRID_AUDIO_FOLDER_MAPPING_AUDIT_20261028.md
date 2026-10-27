# GRID audio / alignment speaker-folder mapping audit

Status: `BLOCKED_AUDIO_SPEAKER_FOLDER_MAPPING`. No AV model prediction or
quality metric was produced. The frozen GRID pilot selection remains intact,
but must not be evaluated until the source mapping is explained by the
provider or an authoritative mapping artifact.

## Inputs and method

- Original GRID alignment split: train SHA-256
  `f6f045931aee53e576fca22efda0bc888a983aed685a30edc38b2918ff156817`,
  validation SHA-256
  `c1e77098327841fadc283bb14813c38de94e07fd16728d9cc3602bc261961e11`.
- Official Zenodo `audio_25k.zip`: 2,607,605,854 bytes, publisher MD5
  `4b3ac37b1a258f55d1eebe657de491a9` matched, SHA-256
  `d89331b1db49770ca0d0c676e75b6e35725ecb48cf35915e8482285af2594134`.
- The audit parses the pinned alignment rows and ZIP member names under
  `audio_25k/<speaker>/<code>.wav`. It reads no audio payload, video, model,
  predictions or sealed audit. It does not infer or apply a speaker remapping.

The result is **18,997 missing same-name audio members among 31,991 alignment
speaker/code rows across 32 speakers**. Thirteen speakers have complete
same-name code coverage; nineteen
speaker folders have no same-name coverage and instead have a unique other
audio folder containing their entire alignment code set. The held-out GRID
validation speakers include four affected labels: `s13 -> s10`, `s18 -> s19`,
`s23 -> s22`, and `s27 -> s26`. This filename overlap is not proof that the
audio in the other directory belongs to the aligned/video speaker, so these
folders are not silently remapped.

The exhaustive result is saved outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/grid-audio-folder-mapping-20261028-attempt-01/mapping-audit.json`,
SHA-256 `e0606f0cfc263c6102766a31c10337fe09dfc6b7f3d6c43efa6dd968fc2b1e84`.
The command is `python scripts/audit_grid_audio_folder_mapping.py --output-dir
C:/CodexArtifacts/pretrained-reuse-r2/grid-audio-folder-mapping-20261028-attempt-01`;
it writes the full mapping report and exits nonzero on this mismatch by design.

## Pilot impact

The pre-media sample is still frozen at 64 quartets / 256 query roles and its
hash remains valid. However, four of its eight validation speaker groups do
not have audio files at the alignment speaker paths, and code-set overlap alone
cannot establish person identity. Do not run the pilot or change its speaker
set to only the four matching labels. The verified `s30.zip` archive remains
preserved (381,061,085 bytes, publisher MD5
`8958f25b49de7a7827a76abbdc7fedb2`, SHA-256
`69e7c2ef14b1bd4f2d0012a8a13fe8e9375c4f11a03c4de9f03f4071f48f12fd`). The
`s4.zip.part` file is an incomplete, unverified 218,996,736-byte download and
must not be used.

Next, establish the official archive-to-alignment speaker mapping or select a
different synchronized source with independently verifiable asset identity.
Until then, GRID is a technical source candidate only and does not close R2.
