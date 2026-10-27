# iNaturalist photo identity metadata join: result

The full compressed iNaturalist photo metadata export was streamed without
retaining the 20,668,099,997-byte source file. It was pinned to ETag
`"c8ded8376313d0a25357209f8cafd0db-40"`; the observed raw SHA-256 is
`033b41cb9e0f28fa2015a66f9bc7e4e95d2ebb1bd96b174e310195eb241888be`.
The pinned scan counted 501,256,126 rows and matched 1,572 of 1,598 target
photo IDs; 26 were absent and six IDs had multiple metadata rows.

The pinned identity join produced 1,570 candidate observations and quarantined
30. It joins photo rows to the pinned observation UUID map, then forms
transitive connected components over observers and photographers before
checking the fixed train/validation split. There are 1,482 train rows across 87
components and 88 validation rows across 18 components. The largest component
contains 481 rows. Per-class validation counts range from 6 to 17 (9, 6, 12,
6, 12, 11, 17, 15 for the eight fixed taxa).

This is **not an admitted image benchmark**. The validation counts are too small
for a useful quality claim; component sizes are highly uneven; source-version
consistency has no independent trusted checksum; and no media bytes were
downloaded or hashed. Observation/photo license declarations and attribution
were not reconciled against every image, and ownership was not verified. No
images were opened, no model was evaluated, and no training split was admitted.
Do not use these metadata rows for quality claims or model selection.

The raw scan summary is
`C:/CodexArtifacts/pretrained-reuse-r2/inat-bird-photo-identity-scan-20261028-attempt-01/summary.json`,
SHA-256 `d02e62f78f1b45d9856bee0c09f3adf754cb0bcab181a77df44496d62ccbb062`.
The join summary is
`C:/CodexArtifacts/pretrained-reuse-r2/inat-bird-connected-metadata-20261028-attempt-01/summary.json`,
SHA-256 `56efe916498593a6f77999093903f7727beef2facc23cbf340dc707ae0114eaf`.
Private row-level IDs remain in the external attempt directory, not this report.
