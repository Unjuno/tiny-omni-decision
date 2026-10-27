# Open Images full train support audit

The fixed [protocol](R2_OPENIMAGES_FULL_TRAIN_SUPPORT_PROTOCOL_20261027.md)
streamed the complete 2,474,671,021-byte pinned train label object without
persisting source rows. Its SHA-256 matched the existing manifest:
`a97b079ccccf635a980bfe4b8180a0c844b4d9c887b2b7bc56535f539af044ea`.
The first attempt failed closed because the audit protocol transcribed the hash
incorrectly; the manifest value was fixed in commit `e1fa97d5` before the
second attempt. Only that second attempt produced a result.

There were 7,337,077 image groups, of which 7,714 have at least one in-scope
verified positive and three distinct in-scope verified negatives. These support
8,553 positive target/image pairs; multiple targets can share an image.

| Class | Eligible target/image pairs | Available as negative in eligible pairs |
|---|---:|---:|
| Toy | 147 | 282 |
| Door | 543 | 1,771 |
| Food | 156 | 4,349 |
| Plant | 787 | 1,779 |
| Sculpture | 730 | 2,349 |
| Vehicle | 317 | 4,434 |
| Clothing | 812 | 2,035 |
| Furniture | 599 | 1,485 |
| Building | 1,769 | 4,959 |
| Window | 718 | 2,533 |
| Animal | 1,975 | 675 |

Negative counts measure available target/negative pairs, not selected
options. All 11 classes have support in both roles, but availability is very
unequal. In particular, if a future four-choice dataset demanded the same
number of positive appearances for every class and the same number of
negative appearances for every class, the 282 available Toy-negative pairs
impose an optimistic upper bound of 94 positive examples per class: each
class would have three times as many negative as positive appearances.
Actual feasible counts can be lower due to image uniqueness, candidate
co-occurrence, rights, asset access and independent split constraints. This
bound is not a proposed sample size and no benchmark has been selected.

This train-label census does not prove individual media rights, usable image
availability, role-balanced construction, split independence or decision
quality. It does not inspect evaluation or sealed audit data. A new task must
freeze a role-aware selection procedure and its constraints before constructing
examples, and must separately audit asset provenance and split integrity.
The existing historical evaluation remains unchanged.

Result: `C:/CodexArtifacts/pretrained-reuse-r2/`
`openimages-full-train-support-20261027-attempt-02/result.json`, SHA-256
`c161e9ef3a62eb96fe73ee36d124ae1fa60531d61646379dac7c7633cd1eec54`.
Run from this worktree with `PYTHONPATH=src;.` and
`python -m scripts.audit_openimages_full_train_support --output <new-json>`.
