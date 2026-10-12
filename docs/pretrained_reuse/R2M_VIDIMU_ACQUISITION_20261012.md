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
