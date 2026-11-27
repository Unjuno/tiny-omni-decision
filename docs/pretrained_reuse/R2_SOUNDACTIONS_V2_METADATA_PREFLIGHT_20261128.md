# SoundActions v2 metadata and task preflight — 2026-11-28

## Disposition

**Useful as a small exploratory audio-video development source; not eligible
to close the R2 general-quality or blind-audit gates.** The publisher's version
2 release is CC BY 4.0, contains 365 aligned audio/video recording references,
and supplies three independent annotation files plus corrected consensus
labels. However, every clip was recorded by one person, the action/object
annotations are mostly unique free-text strings, no official split is
published, and the same 365 labels were already used in a published five-fold
experiment. VideoPrism's public-video pretraining overlap with these source
videos is UNKNOWN. No audio, video, or image payload was downloaded, and no
model was trained or evaluated.

## Pinned release and actual files inspected

- Zenodo record/version: [`10.5281/zenodo.23971067`](https://zenodo.org/records/23971067),
  version 2, published 2026-11-05. The corresponding record ID is `23971067`.
- The record declares CC BY 4.0 for the dataset. The downloaded publisher
  `LICENSE.txt` grants sharing and adaptation, including commercial use, with
  attribution, a licence link and change notice. This clears the repository's
  source-license allowlist for the released dataset; it does not establish
  that any pretrained model has seen or has not seen the recordings.
- Downloaded only `overview.csv` (126,126 bytes), `statistics.json` (857),
  `SHA256SUMS.txt` (118,025), `sound-actions-labels.zip` (58,315), `README.md`
  (8,194), and `LICENSE.txt` (345). Local MD5 values match the version-2
  Zenodo file listing. Their SHA-256 values are in
  [`soundactions-v2-source-preflight-20261128.json`](../../manifests/pretrained_reuse/soundactions-v2-source-preflight-20261128.json).
- The labels archive contains the three per-rater CSVs, corrected consensus
  files, the master table, and a published correction receipt. I read the CSV
  data as records and did not execute the included third-party Python script.
- No MP4, WAV, JPG, YouTube stream, or OSF copy was fetched. The record lists
  approximately 2.29 GB of media, while this metadata-only attempt wrote under
  0.3 MB. No split, media-byte integrity, decoder, or timestamp-sync check has
  been claimed.

The machine-readable audit output is stored outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\soundactions-v2-preflight-20261128-attempt-02\metadata-audit.json`
(SHA-256 `af7b3facaf6402fd8e730473370f098a8f9eafcff48fd4ac2793ffc786bc033e`).
It contains aggregate counts and input hashes, not raw titles, URLs, or record
IDs.

## Data quality findings

The metadata describes one recording each day during 2022, all by the same
recordist, using a Samsung S21 Ultra and a Røde VideoMic Me-C. It lists 365
unique record IDs and unique WAV/MP4/JPG filename references. The publisher
describes those 365 recordings as real-world sound-producing actions; each
record can contain a sequence of operations, so a clip is not necessarily one
atomic action.

The three raters independently annotated every recording. For the four-way
PerceptionType target, 334 items have a majority and 31 are `unknown`; only 158
items are unanimous. The published free-marginal kappa is 0.46. Majority class
counts are 126 impulsive, 125 iterative, 83 sustained, and 31 unknown. Keep the
unknown cases in any future evaluation rather than dropping disagreements.

The three raters supplied 1,095 action fields with 944 distinct normalized
full-field values, 848 of them singletons. Object fields have 940 distinct
values, 808 singletons. These are sparse free-text annotations rather than a
stable, repeated action/object taxonomy. The publisher's v2 correction receipt
records 39 corrected cells across 35 items; using v2 is necessary because the
paper and pre-August-2026 OSF copies have stale consensus cells.

The publisher does not provide a train/validation/test split. The 2026 paper
reports five-fold experiments over this same 365-item source. Its all-adapter
PerceptionType results were 0.66 accuracy for audio-only, 0.60 for audio+video,
and 0.48 for video-only. These are results for DG-SCT under the paper's folds,
not this repository's frozen Path A model, but they warn that the target may be
mostly audio-solvable and that this source cannot establish an AV gain without
matched controls.

## Product and evaluation limits

1. It is a useful *development* probe for whether aligned sound and visible
   actions help a compact readout predict perceptual sound type. It is not a
   general audio/video Decision benchmark and does not measure text, image,
   sensor, arbitrary-option reasoning, or 95–98% quality.
2. One recordist means all results are conditional on that recordist and
   capture setup. Three annotators provide a useful human-label check but do
   not provide broad population coverage.
3. The public split is absent, and the labels have been used in a published
   five-fold experiment. A project-held-out split could still support a
   development comparison, but it must not be called a blind external audit.
4. The pinned VideoPrism checkpoint was pretrained on a very large public
   video corpus that includes YouTube-related sources. The exact overlap with
   SoundActions' publicly listed videos is UNKNOWN. A score using that tower
   could not be called leakage-free without a corpus-level exposure check.
5. The full release is small enough for local storage, but that alone does not
   justify downloading or training. The next gate is to define a frozen split
   and an exact task/contrast that addresses the source's audio dominance and
   single-recordist limitation. If it cannot support those controls, stop at
   this preflight.

No pre-existing corpus, run, checkpoint, branch, PR, or sealed audit was
changed. A dedicated source-audit branch was created for this metadata-only
preflight.
The local RTX 3080 remained occupied; no GPU work was started. No paid compute
or external account access was used.

## Reproducible metadata audit

```text
py -3 scripts/audit_soundactions_metadata.py --overview <overview.csv> --labels <sound-actions-labels.zip> --output <metadata-audit.json>
py -3 -m pytest tests/test_soundactions_metadata_audit.py -q
```

The auditor checks the 365 unique IDs, one-to-one filename references, rater
coverage, consensus agreement, correction receipt and sparse free-text counts.
It never prints raw titles, URLs or IDs and never reads media bytes.

## Primary references

- [Zenodo version 2 record and file listing](https://zenodo.org/records/23971067)
- [Guo, Tørresen & Jensenius (2026), SoundActions paper](https://doi.org/10.5334/tismir.223)
- [Official VideoPrism repository and pretraining description](https://github.com/google-deepmind/videoprism)
- [Google Research VideoPrism pretraining overview](https://research.google/blog/videoprism-a-foundational-visual-encoder-for-video-understanding/)
