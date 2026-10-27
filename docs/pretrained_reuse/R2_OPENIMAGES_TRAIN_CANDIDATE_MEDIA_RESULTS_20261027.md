# Open Images candidate media: bounded acquisition

The [fixed protocol](R2_OPENIMAGES_TRAIN_CANDIDATE_MEDIA_PROTOCOL_20261027.md)
attempted all 258 page-declaration-matched train candidates: 257 JPEGs were
saved and decoded, and one returned HTTP 410. Total saved media was 25,562,755
bytes. The new external directory is separate from earlier caches. No new
validation images, validation labels, model or checkpoint were accessed.

| Original target | Saved candidate images |
|---|---:|
| Toy | 23 |
| Door | 23 |
| Food | 13 |
| Plant | 26 |
| Sculpture | 15 |
| Vehicle | 25 |
| Clothing | 27 |
| Furniture | 28 |
| Building | 27 |
| Window | 24 |
| Animal | 26 |

Against 167 and 168 previously saved validation images respectively, there
were zero normalized RGB SHA-256 matches and zero perceptual hash pairs with
Hamming distance <=4. An additional read-only comparison against 165 saved
old train images also returned zero such pairs. No pairs within the 257 new
images met that same threshold. These are limited hash-based screens, not
proof against crops, altered images, related scenes or overlap with unseen
validation assets.

The page declarations were verified separately; ownership is not independently
verified. These images remain candidates pending the complete project rights,
split, content and role-balance gates; they are not a selected training set.
No training or accuracy evaluation was performed.

External artifact directory:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-in-scope-train-media-20261027-attempt-01/`.
Raw SHA-256 for `summary.json`:
`0d13b7b716607a5e95f79e8d09c26a27276bab4ffd131aaf5f08fe17d7a72adc`;
for `private-media-manifest.jsonl`:
`7ff4b8bda635812ed404adf06b279e55a96838f18407e39ab80e9e659f59a9a7`;
for `private-duplicate-warnings.json`:
`a5338d955b09046ec0b16f3a9625b7955c763aae07dc722e474e6078745f932f`.
The manifest records individual JPEG hashes. The summary's manifest hash is
for the LF text payload and differs from the Windows raw-file hash above.
