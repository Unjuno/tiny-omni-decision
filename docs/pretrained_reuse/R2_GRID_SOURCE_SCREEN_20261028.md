# GRID audiovisual corpus: preliminary source screen

The [Zenodo source record](https://zenodo.org/records/3625687) describes
34,000 fixed-grammar utterances from 34 talkers, paired speech audio and face
video, with word-level alignments. It explicitly notes video for one talker is
unavailable. This could be a controlled speech/vision diagnostic, but the
record's visible rights section does not specify a license. Open download is
not by itself permission to train or redistribute. Status is **REVIEW**; no
corpus media or alignment files were fetched or admitted.

The [original Sheffield distribution page](https://spandh.dcs.shef.ac.uk/gridcorpus/)
explicitly describes the corpus and transcriptions as freely available for
research use, but does not state a commercial license or detailed consent
scope. A 2021 paper cites the Zenodo copy as CC-BY-4.0, but that secondary
statement does not resolve the missing visible license on the current source
record or establish the terms of the Sheffield distribution. Continue to treat
the exact intended distribution and use as REVIEW pending primary evidence.

### Zenodo license and Lombard relationship

The primary `https://zenodo.org/api/records/3625687` response was subsequently
retrieved: `metadata.version` is `1.0`, `metadata.access_right` is `open`, and
`metadata.license.id` is `cc-by-4.0`. This supplies a license statement for
the Zenodo distribution, correcting the prior unresolved license status. The
[European Language Grid catalog](https://live.european-language-grid.eu/catalogue/corpus/7769)
independently describes the same DOI and license. CC-BY attribution and
participant/data conditions remain applicable; the per-speaker Sheffield
distribution has its own research-use wording and has not been shown to carry
identical terms.

The [Lombard Grid authors](https://spandh.dcs.shef.ac.uk/avlombard/) state that
their 54-speaker corpus uses the same sentence format, but distinct sentence
sets from original GRID. This is promising for a controlled transfer probe
relative to the existing Lombard baseline. The statement does not establish
speaker independence or that two modalities are necessary, and exact audio,
video and alignment identity have not been checked locally. Original GRID is
not yet an admitted task or a blind quality result. Freeze source and split
rules before decoding, and verify actual sentence/source and speaker overlap;
do not equate matching numerical speaker IDs with matching individuals.

### Derivative index versus existing Lombard train codes

A preliminary exact-code check used the public derivative
[`SilentSpeak/EGCLLC` alignment index](https://huggingface.co/datasets/SilentSpeak/EGCLLC/blob/b61192d1a5790843425862111a0eb870d911e8a9/alignments.txt),
not the still-pending Zenodo archive. Its 32,827 index paths contain 32,827
distinct six-character utterance codes. The existing Lombard train manifest
contains 4,080 rows and 1,968 distinct actual codes. Seven distinct codes match,
affecting eight Lombard train rows. This is a concrete warning against assuming
the author's distinct-sentence statement proves zero overlap for these particular
files. It does not establish whether the derivative index accurately represents
the original, whether alignment contents match, or whether speakers overlap.
No validation or media was read. Compare the verified original alignment archive
and inspect the specific code mapping before declaring source/content isolation.

The aggregate output is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-lombard-index-20261028-attempt-02.json`,
SHA-256 `1e7265574dfbf80af3ff75c53e783fd32621deae21bad14070eafd3d6ad2a921`.
It pins the index, train manifest and script hashes. The initial attempt used
an equivalent script before a formatting-only correction and remains preserved.
Two focused tests and Ruff passed; no original alignment or media was fetched
by this comparison.

### Original and derivative alignment contents

The original Zenodo `alignments.zip` was downloaded without media; its
20,091,485 bytes match the publisher MD5
`df892a5adea6595339b331cc065501f9`, and its locally computed SHA-256 is
`161bcf2548797e6cf0116541e8c76fdb34e014a1dd4d33d2909cb12e588232ee`.
The separately published derivative alignment TAR matched its pinned 67,338,240
byte size and SHA-256
`40abb10300984775e6465b6ab6665231c8d81f3a9af559622d6f48a48d902f11`.
Both archives independently contain the seven matching codes; all eight
associated Lombard training rows have exactly matching six alignment words
after converting numeric target labels to their written form. macOS sidecar
entries in the ZIP were excluded by a strict path pattern. This confirms
**transcription overlap**, not identical recordings or speaker identity.

The original-content receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-lombard-original-content-20261028-attempt-01.json`,
SHA-256 `15fe0ef426b92efe32f0f874ebee88ca4b26611447cb5655dde1d5c068394c5c`;
the derivative-content receipt is `grid-lombard-content-20261028-attempt-01.json`,
SHA-256 `88844668ad29bb8666480ea9c215e294cfcf940deb9b1bf31d7037514ab7d3ca`.
Both compare only existing Lombard train labels and alignment text, without
media or validation. Two focused tests and Ruff passed. Any transfer cohort
must exclude or separately label these overlapping utterances and audit speaker
identity; the remainder is not automatically independent.

### Original alignment coverage after training-text exclusion

Scanning only original ZIP member names found 34,000 valid, distinct utterance
codes across 34 speakers. After excluding the seven codes in the existing
Lombard train manifest, 33,993 alignment rows remain, with 998–1,000 per
speaker. Of these, 1,000 are for speaker 21, for whom the publisher says video
is unavailable; the remaining 33 speakers account for 32,993 alignment rows.
These are alignment candidates, not verified media pairs or independent people.
The check did not compare against previously observed Lombard validation or
prove a final benchmark split; a future task must preserve its own holdouts and
audit all applicable source/content identities before evaluation.

Aggregate receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-candidate-coverage-20261028-attempt-01.json`,
SHA-256 `5f27f9cf0cd464a650c60ad904780b958f3d652a049d7efbd18773500267ce56`.
Two tests check exclusion, the missing-video group and duplicate-code rejection;
Ruff passed. No audio or video was opened during this scan.

The fixed vocabulary, scripted sentences, limited speakers and potential
audio-only or lip-only shortcuts also preclude treating a high score on this
corpus as general Omni Decision quality. Before use, independently establish
the permitted license and participant/data conditions, verify synchronization,
keep speakers and utterance content appropriately separate, and define
missing-modality and mismatched-input controls. The record reports publisher
MD5 checksums, but these do not replace locally verified SHA-256 receipts.
