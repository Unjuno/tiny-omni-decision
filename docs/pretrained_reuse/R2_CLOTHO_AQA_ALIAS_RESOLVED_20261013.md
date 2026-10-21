# Provenance-reviewed alias for source16854

Retrieved the actual source page
https://freesound.org/people/deegnot/sounds/16854/ . It identifies the title as
`souffle_me?tallique.wav`, uploaderdeegnot, duration22.8117s, and links its official
preview at https://cdn.freesound.org/previews/16/16854_36395-lq.mp3 .
Original Clotho and AQA metadata use underscore at the corrupted character;
the AQA ZIP uses a dot. No other filename substitutions are admitted.

Downloaded the candidate ZIP member with CRC verification and the public
source-ID-specific preview. Decoded both to mono2kHz with PyAV. Full candidate
correlation against preview: .934503273, offset155samples. Candidate44,977samples,
preview45,624samples. Eleven successive2s candidate segments align at offset155
plus their starting position, except the first at163; correlations .9218–.9574.
This coherent whole-recording and local alignment supports a provenance-reviewed
alias together with the specific source title/uploader. It is not byte equality,
a general matching classifier, or an automatically passed .99 duplicate gate.
Preview compression and different duration remain explicit limitations.

Admit only source16854/uploaderdeegnot/name`souffle_me_tallique.wav` to archive
member`audio_files/souffle_me.tallique.wav`, requiring acquired candidateSHA
`5fb420a29ce8fe73fcd035f2b8366d20f32e095deac57f9328d82f2e848c9a0a`.
The helper resolves the name; full acquisition must also enforce this hash.
All856 frozen source IDs now resolve in the index. No question, label, source ID,
split assignment or frozen manifest byte was changed. Three tests passed for
exact alias and rejection of conflicting source/uploader; Ruff passed.

Evidence: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-alias-waveform-01/`
contains source.html, preview.mp3, candidate.wav and comparison reports/hashes.
The adjacent JSON preserves the full/segment measurements. The preview is
identity evidence only and must not become a training example or replacement
for the licensed ZIP WAV. Next: full bounded acquisition and waveform overlap
checks, followed by the fixed audio-only/joint comparisons.
