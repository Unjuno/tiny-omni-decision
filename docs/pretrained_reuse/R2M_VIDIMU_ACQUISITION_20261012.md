# VIDIMU: acquisition underway, synchronization convention inspected

Pinned Zenodo record8210563 declares CC-BY4.0. dataset.zip is367,226,494bytes,
publisherMD5 `7182e1920da1091cb82ec8279afbe773`. The API file endpoint supports
ranges but returns no ETag, so attempt01 stopped before archive inspection.
Attempt02 downloads this one archive and verifies full size/MD5 before opening
it; the recorded local SHA256 will supplement publisher MD5. No full video or
analysis archives are being downloaded. Acquisition is not yet a verified result.

The inspected upstream synchronization notebook at
`twyncoder/vidimu-tools@649fdf07513fddcf5b0462a1bd888a5b13319c9c`
uses Subject/Activity/File/Type/CutFrames metadata from analysis/videoandimusync.
It enumerates16 subjects (S40,S41,S42,S44,S46–S57), with raw IMU50Hz versus
video30Hz. It trims raw/mot/csv text records but explicitly skips mp4 modification.
Therefore corrected text files must not simply be paired with frame0 of original
video. Need actual CutFrames values and retained timestamp conventions first.
The code is inspected, not copied or executed; upstream software is GPLv3 while
the dataset has its own CC-BY4.0 declaration.

Current acquisition directory:
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-source-audit-02`.
Partial dataset.zip must not be trusted before script success/result.json.
Code snapshot: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-code-audit-01`.
No payload contents, quality evaluation, split assignment or training completed.
Do not restart a live download based on a polling timeout.

Sources: [dataset](https://zenodo.org/records/8210563),
[upstream tools](https://github.com/twyncoder/vidimu-tools).

## Verified completion (13:45 JST)

Attempt02 exited successfully. Exact archive size and publisher MD5 matched.
Local SHA256:
`605b695dcec5c2e7c4fd9a81dda22b60049697063d9eda150a637166813b79c2`.
The archive contains 2,076 entries, including original and corrected sensor/
video-derived records, but no `infoToSync.csv` or README. No sensor records have
been decoded. The corrected directory is not assumed to establish alignment.

The separately pinned 169,608,328-byte `analysis.zip` is now being acquired by
`scripts/audit_vidimu_analysis.py`, using the metadata already preserved from
record8210563 and publisher MD5 `6c937c7122fee77ff1ce306175cb26c7`.
Output: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-analysis-audit-01`.
Only archive inventory and `infoToSync.csv` will be read after full verification.
This is acquisition in progress, not a verified synchronization result.
See the separate synchronization-code audit for the concrete offset ambiguity.
