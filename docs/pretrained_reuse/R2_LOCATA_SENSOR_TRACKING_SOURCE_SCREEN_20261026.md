# LOCATA development corpus: audio + moving-sensor task screen (2026-10-26)

## Disposition

**Promising for a narrowly scoped audio + sensor Decision probe; not admitted for model fitting yet.** LOCATA Task 6 is the closest source found in this screening round to the optional sensor path in R2-M: multiple moving talkers are recorded by moving microphone arrays, and the corpus provides synchronized source and array positions/orientations. A decision over global-coordinate source candidates could require the acoustic evidence and array-pose stream together. That task definition and the sufficiency controls still need to be validated from the development annotations; co-recorded signals alone are not proof that both inputs are necessary.

This is an audio + sensor candidate, not an audio + video result and not a substitute for the required text/image/audio/video task battery. No model, readout, sensor feature, or accuracy metric was produced. The official evaluation archive and all sealed-audit material remain untouched.

## Official source and rights screen

- Dataset: LOCATA Challenge final development/evaluation release, Zenodo DOI [10.5281/zenodo.3630471](https://zenodo.org/records/3630471), record ID `3630471`.
- The live Zenodo API reports an open record and `metadata.license.id=odc-by`. The official corpus page says the data corpus and its VCTK speech material are distributed under Open Data Commons terms. The official dataset page links the Zenodo release and notes a corrected final development database.
- The official description defines Task 6 as tracking multiple moving talkers using moving microphone arrays. It describes four array configurations and ground-truth position, trajectory, and orientation from an optical tracking system.
- The data bundle also includes MATLAB reader/evaluation software. Its license is not established by the data license; this project has not used that code.
- This is a scoped source screen, not a comprehensive legal or privacy review. Before any fit, inspect the development documentation for participant, recording, timestamp, and target semantics, and use a project-authored parser.

Primary references: [official corpus description](https://www.locata.lms.tf.fau.de/corpus/), [official dataset page](https://www.locata.lms.tf.fau.de/datasets/), [pinned Zenodo record](https://zenodo.org/records/3630471), and [challenge paper](https://doi.org/10.1109/TASLP.2020.2990485).

## Metadata-only archive inspection

The Zenodo API reports `dev.zip` as 6,207,354,195 bytes with declared MD5 `d5a5417c3f6b2ed0e43581dd06f504e6`; `eval.zip` is 13,045,105,034 bytes. The directory listing parsed only ZIP central-directory metadata, using four HTTP 206 range requests (117,306 response bytes total). Separately, a 1 MiB leading-range smoke and the cancelled transfer received raw leading bytes from `dev.zip`; no file was extracted, decompressed, or semantically inspected. No `eval.zip` content byte was fetched.

The directory contains 921 entries (816 files: 639 `.txt`, 177 `.wav`) across `task1` through `task6`. Task 6 contains 12 task/recording/array paths and four literal array IDs. Its compressed member totals are:

| Archive path array ID | WAV/TXT members | Compressed bytes | Uncompressed bytes |
| --- | ---: | ---: | ---: |
| `benchmark2` | 42 | 297,850,711 | 2,008,340,414 |
| `dicit` | 42 | 1,152,689,559 | 2,228,888,844 |
| `dummy` | 42 | 167,779,124 | 1,389,800,377 |
| `eigenmike` | 42 | 443,457,769 | 3,479,315,679 |

These are ZIP member-size totals, not inferred independent examples. A follow-up metadata-only audit of the ZIP central directory confirms only three recording IDs per array configuration: `recording1`–`recording3` for `benchmark2`, `dicit`, and `eigenmike`, and `recording4`–`recording6` for `dummy`. These are at most six recording groups overall, not six proven independent speakers or scenes. Array variants of a recording must remain in the same split. Any repeated speaker, sentence, or session across recordings must be audited before defining train and development validation.

**Decision: do not acquire or use LOCATA as a quality-evaluation or fitting source in this workstream.** The current R0 sensor gate calls for at least 200 independent subjects/scenes unless a low-power protocol is explicitly justified. The central directory gives at most six recording groups, while speaker/session independence is not established; this is far below that gate. A mechanics-only parser smoke could be reconsidered separately, but it would not count as task-quality evidence. No Task 6 member payload was requested or read in this follow-up, and `eval.zip` remains untouched.

## Bounded transfer attempt and next gate

A full `dev.zip` transfer was started to a new external directory and stopped by this task after 9,269,248 bytes when measured throughput remained about 0.27 MB/s, implying roughly 6.1 hours for the remainder. The incomplete file is preserved and explicitly named as partial; it is not treated as an archive or dataset artifact. A 1 MiB range request confirmed the server supports verified `206 Partial Content`, so a later bounded extraction can request only selected Task 6 member byte ranges rather than transfer the whole archive. No GPU, model training, or evaluation ran; no paid compute was used.

External metadata receipts, outside Git:

- API record snapshot: `C:\CodexArtifacts\pretrained-reuse-r2\locata-dev-source-screen-20261026-attempt-01\zenodo-record.json`, SHA-256 `BC74E5B55F177AA013E5653D405EA8B84EFB9DD2E166FCBA70205DCA22BD3540`.
- Partial transfer receipt: `C:\CodexArtifacts\pretrained-reuse-r2\locata-dev-source-screen-20261026-attempt-01\partial-download-receipt.json`, SHA-256 `A3B002FF0A3E655FAEDCF17E326FCB94CB998865265DD32593A26A7C5D63F256`.
- ZIP central-directory listing: `C:\CodexArtifacts\pretrained-reuse-r2\locata-range-preflight-20261026-attempt-01\listing.json`, SHA-256 `649B378182CC0E298DE9545D8BF77EB46AFBD5A122F59C709794951423AC5182`.
- Range-listing driver SHA-256: `0262892B4317CFB05A0C7BCD1697DA8D811BC573862B3C2C7826CCFA8E2A2184`.
- 1 MiB range smoke file: C:\CodexArtifacts\pretrained-reuse-r2\locata-range-preflight-20261026-attempt-01\first-megabyte.bin, SHA-256 22B6D925257862E5DA4DE26AAC402402E88F5BCBA74E342B6D93FD8AE87E905D (raw leading archive bytes only; not extracted).
- The incomplete development download is `C:\CodexArtifacts\pretrained-reuse-r2\locata-dev-source-screen-20261026-attempt-01\dev.zip.partial-incomplete`; it contains no complete ZIP central directory and must not be consumed.

### Next executable gate

Before any training:

1. LOCATA is not on the quality-evaluation path under the current independent-group gate. Do not spend transfer or GPU budget on it for the 95–98% objective.
2. Reconsider only if a materially broader, rights-cleared source supplies enough independent speaker/session groups, or if the project explicitly approves a low-power mechanics study. Keep any such study separate from the four required modalities and from product-quality claims.

No quality, temporal-generalization, product-accuracy, or 95–98% claim follows from this source screen.
