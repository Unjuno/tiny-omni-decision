# Open Images: symmetric-choice train-prefix feasibility

A [fixed train-only protocol](R2_OPENIMAGES_SYMMETRIC_PREFIX_PROTOCOL_20261027.md)
read the first 67,108,864 bytes of the pinned 2,474,671,021-byte human-label
object using the recorded GCS generation, ETag and exact HTTP 206 byte range.
The final partial image group was excluded. The received prefix SHA-256 is
`43b0c2af6459a85000434d2d0e8244290d562296374d52cf79692920a6a5847f`.
This is a lexicographic prefix, not a random or representative sample; the
full-object SHA-256 was not recomputed. No validation, test, media or model was
opened and no individual image license was established.

On 8,216 complete train image groups, the number of images per positive
class with **at least three distinct explicitly verified negative labels in
the same frozen 11-class scope** was:

| Target | Positive image groups | With >=3 in-scope explicit negatives |
|---|---:|---:|
| Toy | 102 | 4 |
| Door | 157 | 11 |
| Food | 178 | 4 |
| Plant | 788 | 31 |
| Sculpture | 191 | 8 |
| Vehicle | 768 | 16 |
| Clothing | 873 | 26 |
| Furniture | 293 | 6 |
| Building | 544 | 31 |
| Window | 323 | 31 |
| Animal | 571 | 30 |

These add to 198 target/image pairs, not necessarily 198 distinct images;
an image can have multiple positive targets. All 11 classes have at least
one such record in this prefix, but four classes have fewer than ten. The
pinned hierarchy check confirms that each of the 11 classes has all other ten
as siblings under the existing builder's rule. It does not imply those labels
are annotated or absent in every image.

Restricting all choices to the target scope would remove the specific membership
shortcut, but **does not establish balanced positive/negative class roles or
remove other option priors**. Nor does this screen demonstrate sufficient
full-corpus coverage, individual Flickr rights, source independence, a locked
new split or model quality. Do not filter historical evaluation to these cases.
Next, perform a frozen full-train-source support and target/distractor role audit
before designing and freezing a new task; disclose any class that cannot meet
coverage instead of dropping it. The 95–98% objective remains unmet.

Result:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-symmetric-prefix-20261027-attempt-01/result.json`,
SHA-256 `863476464e1bf6283fa8b8498f63c356c9a39f4d0be070ef3f05ebdbf733c736`.
The result pins the exact whitelist, source generation and prefix, code and
protocol. The independent hierarchy check is
`python -m scripts.audit_openimages_scope_siblings`; the network prefix screen
is `python -m scripts.audit_openimages_symmetric_prefix --output <new-json>`
with `PYTHONPATH=src;.`. The latter re-reads the 64 MiB network range.
