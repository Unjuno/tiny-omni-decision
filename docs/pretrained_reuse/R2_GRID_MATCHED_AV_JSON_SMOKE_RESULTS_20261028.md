# GRID matched audio/video JSON smoke: result

This is a single fixed-model interface smoke, not a quality evaluation. The
audio archive completed from the pinned Zenodo GRID v1.0 URL at 2,607,605,854
bytes. Its publisher MD5 matched `4b3ac37b1a258f55d1eebe657de491a9` and its
SHA-256 is
`d89331b1db49770ca0d0c676e75b6e35725ecb48cf35915e8482285af2594134`.
The 16 preselected `s1` WAV files were found in the archive, decoded as 25 kHz,
resampled to the frozen audio path's 16 kHz input, and checked for nonempty,
finite samples. The decode receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-s1-audio-decode-20261028-attempt-01/summary.json`,
SHA-256 `fc755d1ffbf458ad97f32788542d54ee8d6aecb9398d2063eea8bd9a28bcfc9d`.

One matched query used the `bbaf2n` audio and four fixed video candidates
`bbaf2n`, `bbaf3s`, `bbaf4p`, and `bbaf5a`. Frozen audio/video features and
the existing correspondence runtime returned a finite four-option JSON with
probabilities summing to one. The model selected `option-1` (probability
0.3876); the matching media was candidate position 0. This mismatch is kept
as an observed output and was not used to tune or select anything.

The runtime receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-s1-matched-av-json-20261028-attempt-01/summary.json`,
SHA-256 `bf6589aa6ff9bbffea2bc1bcb527e428e4e232216bf74b2e3cd6bbf7559ab1c3`.
The trial used CPU only, performed no training, opened no sealed audit, and
made no accuracy claim. It is a single talker and a narrow, unbalanced smoke;
cross-corpus speaker independence remains unknown. The chosen candidate and
probability are not evidence of useful general audio/video fusion.

The first invocation stopped before inference because its temporary iterator
paired 16 manifest rows with four selected video byte strings under strict
length checking. The corrected invocation limited both candidate lists to the
same first four rows and completed. The persistent model, archive and
checkpoint files were read only.
