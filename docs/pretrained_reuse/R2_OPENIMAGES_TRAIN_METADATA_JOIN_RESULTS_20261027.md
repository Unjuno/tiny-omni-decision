# Open Images train metadata joined support

The [fixed protocol](R2_OPENIMAGES_TRAIN_METADATA_JOIN_PROTOCOL_20261027.md)
streamed and verified both pinned source objects and joined the image IDs.
All 1,743,042 unique metadata IDs declare the train subset and CC BY 2.0, and
each has a group in the annotation object. A group can have no verification
labels; group presence alone does not imply annotation support. Among those
train IDs, 3,377 distinct images support 4,007 target/image pairs with at
least three distinct explicitly verified in-scope negatives.

| Class | Eligible positive target/image pairs | Available negative target/class pairs |
|---|---:|---:|
| Toy | 117 | 210 |
| Door | 511 | 1,649 |
| Food | 76 | 373 |
| Plant | 588 | 1,469 |
| Sculpture | 116 | 122 |
| Vehicle | 238 | 517 |
| Clothing | 577 | 1,739 |
| Furniture | 552 | 1,372 |
| Building | 408 | 2,441 |
| Window | 650 | 2,340 |
| Animal | 174 | 191 |

These replace the previous source-object-wide figures for train-image
planning; the earlier 7,714 image count was not metadata joined. Under exact
equal positive counts and exact equal negative counts across all 11 classes,
the 122 available Sculpture-negative pairs give an optimistic upper bound of
40 examples per class: four-choice construction requires three negative roles
for each positive role. This is only a necessary bound, not a proven feasible
sample size. Image uniqueness, co-occurrence, individual rights, attribution,
feature extraction, splits and other option-only shortcuts can lower it.
Do not downsample a held-out evaluation to obtain this symmetry.

The source declares CC BY 2.0 but individual license and ownership remain to
be checked; no selected IDs or image media were saved. No validation or test
data, model or checkpoint was accessed. The joined aggregate is
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-train-metadata-join-20261027-attempt-01/result.json`,
SHA-256 `45288d892afd8207315f0a2ea73a134d01321b16e7fe4fb046671ee3e2b3ecb7`.
It pins both source hashes, scope, script and protocol. Run the module
`scripts.audit_openimages_train_metadata_join` with a new output path and
`PYTHONPATH=src;.` to reproduce.
