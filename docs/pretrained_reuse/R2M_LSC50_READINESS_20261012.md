# LSC50: explicit paired-source license and archive inventory

[Publisher record27383016 version1](https://figshare.com/articles/dataset/27383016/1)
declares CC-BY4.0 for the video/IMU/landmark dataset. Author attribution: Christian
Cifuentes-De la Portilla, Luis Felipe Giraldo, Brayan David Solorzano Quevedo.
This is a stronger rights candidate than sources with unspecified sensor terms.
It does not yet establish synchronization correctness or useful fusion quality.

Actual versioned API files:

| File | Bytes | Publisher MD5 |
|---|---:|---|
| IMU.zip |1,608,738,643|f94fc68d1ec5da41f9074e77d017a9c8|
| VIDEOS.zip |6,247,307,413|26003107442257c7947352b18e3c2597|
| LANDMARKS.zip |2,770,068,538|996d3e9ee1d3fb39866ce395efd23214|

Only IMU/video central directories were read using exact HTTP206 ranges and
If-Match identity checks. IMU has22members, including timestamp XLSX and5
volunteer-info files; video has4004members including body-color clips. Names
alone do not prove participant/trial/gesture mapping. Whole-file MD5 values are
publisher metadata, not locally verified downloads. Landmarks were not fetched.

Attempt01 stopped on403 from HEAD against a signed GET URL; preserved metadata.
Attempt02 used a1byte GET range then bounded ZIP reads and completed. No sensor
array, video recording or label-sheet content was opened. Local metadata,
headers and inventories: `C:/CodexArtifacts/pretrained-reuse-r2/lsc50-source-audit-02`.
[Audit result](R2M_LSC50_READINESS_20261012.json).

Next: verify published naming and clock conventions, inspect only metadata files,
freeze participant-level splits before media/labels, then retrieve one designated
training clip and its matching sensor interval with CRC/SHA verification. Avoid
downloading all10.6GB into the limited local workspace. Do not invent a mapping
from filename order or conflate derived landmarks with independent IMU evidence.
Still no real VS/AVS learning or quality result; target Omni goal remains open.
