# Open Images train-metadata joined support audit

The previous full annotation-object census did not join official train image
metadata; its counts must not be represented as train-metadata counts. To fix
this, stream both full pinned objects with exact generation, ETag, byte length
and SHA-256. The train metadata is `train-images-boxable-with-rotation.csv`,
generation `1526567626302755`, ETag `a4ac0bcedb5c2df4d1b1230dc6b41b8f`,
length 638407721, SHA-256
`05f3d68dbbb03728d1a37e51479f4f35c062b871e1a6cae8c4cefbe0e0c80ed0`.
Require unique IDs, `train` subset, and CC BY 2.0 metadata; keep IDs only in
memory. Join these IDs to the pinned full annotation object and the previously
frozen 11-class scope. Count eligible unique train images, positive target/image
pairs, and available target/negative co-occurrences exactly as in the previous
audit. Persist aggregate counts only, never individual IDs, media or author
records. Do not read validation labels, predictions or model weights.

The join verifies declared metadata, not individual license or ownership. It
does not create examples, establish independent splits, prove class-role balance,
or support model quality claims. Freeze any future candidate-selection rule
before extracting identifiers; retain historical evaluation unchanged.
