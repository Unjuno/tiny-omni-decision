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

### Aggregate role-flow check (2026-10-28)

The pinned result's 11-by-11 available-negative-pair counts were also checked
using a bipartite maximum-flow relaxation (NetworkX 3.6.1). For each target
class, the source edge has capacity `3*n`; for each distractor class the sink
edge has capacity `3*n`; each distinct target/distractor edge has capacity
`min(n, available_negative_pairs[target][distractor])`. Self-edges are omitted.
For `n = 20, 30, 35, 40`, maximum flows were respectively `660, 990, 1155,
1320`, matching the required `33*n` in every case. Thus the aggregate pair
counts do not disprove equal role counts at the previous bound of 40 per class.
This relaxation does not enforce which three negative labels coexist on an
image, image uniqueness, source splits or rights. It is not evidence that 440
valid examples can actually be constructed or that option-only leakage is gone.

A separate, already completed [full-train quartet analysis](R2_OPENIMAGES_FULL_TRAIN_QUARTET_RESULTS_20261027.md)
did enforce within-image label co-occurrence and unique images and found 410
disjoint same-option quartets. Its class totals are highly unequal (for example,
Sculpture has nine and Window has 331), so it does not establish the equal
40-per-class construction considered above. The greedy search is not an
optimality proof. Rights, duplicates and source-disjoint evaluation remain open.

The source declares CC BY 2.0 but individual license and ownership remain to
be checked; no selected IDs or image media were saved. No validation or test
data, model or checkpoint was accessed. The joined aggregate is
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-train-metadata-join-20261027-attempt-01/result.json`,
SHA-256 `45288d892afd8207315f0a2ea73a134d01321b16e7fe4fb046671ee3e2b3ecb7`.
It pins both source hashes, scope, script and protocol. Run the module
`scripts.audit_openimages_train_metadata_join` with a new output path and
`PYTHONPATH=src;.` to reproduce.
