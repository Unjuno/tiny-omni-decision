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

The fixed vocabulary, scripted sentences, limited speakers and potential
audio-only or lip-only shortcuts also preclude treating a high score on this
corpus as general Omni Decision quality. Before use, independently establish
the permitted license and participant/data conditions, verify synchronization,
keep speakers and utterance content appropriately separate, and define
missing-modality and mismatched-input controls. The record reports publisher
MD5 checksums, but these do not replace locally verified SHA-256 receipts.
