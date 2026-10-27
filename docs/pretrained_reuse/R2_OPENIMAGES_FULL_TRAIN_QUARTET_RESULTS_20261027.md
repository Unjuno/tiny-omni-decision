# Full train metadata: same-option quartet feasibility

The [fixed train-only protocol](R2_OPENIMAGES_FULL_TRAIN_QUARTET_PROTOCOL_20261027.md)
verified and joined both full pinned source objects. Of 3,377 eligible train
images, a deterministic greedy procedure constructed **410 disjoint quartets**
using 1,640 unique images; 1,737 eligible images were unused. The procedure
found 318 distinct option sets, of which 100 had at least one positive example
for each of the set's four labels. Each selected quartet presents the same
ordered four options and has each option correct once, so an option-only
predictor seeing no image or image identity scores 25% within each quartet.

| Target | Images in the greedy quartets |
|---|---:|
| Toy | 11 |
| Door | 239 |
| Food | 21 |
| Plant | 185 |
| Sculpture | 9 |
| Vehicle | 75 |
| Clothing | 220 |
| Furniture | 210 |
| Building | 325 |
| Window | 331 |
| Animal | 14 |

All 11 classes appear, unlike in the earlier 257-candidate subset. The counts
remain highly imbalanced. This greedy result demonstrates that 410 disjoint
quartets exist under the annotation constraints; it is not a proof of the
maximum possible count or a balanced 11-class benchmark. Its images were not
selected for acquisition and individual IDs were not saved. Individual license,
ownership, media availability, visual duplicates, author isolation and an
independent evaluation split remain unverified. No model was run and no
validation, test or audit labels were read. Preserve existing evaluation sets.

The aggregate result is
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-full-train-quartets-20261027-attempt-01/summary.json`,
SHA-256 `c294b4d0fb8622efc89f93a47f6d8d68489090c53f9280cca23a7d0d6388baf8`.
Source, scope, procedure and code hashes are recorded in the summary.
