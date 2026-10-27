# Source-disjoint train quartet candidate screening

The [fixed protocol](R2_OPENIMAGES_QUARTET_CANDIDATE_PROTOCOL_20261027.md)
verified both full source hashes, excluded prior train candidate identities and
authors, and excluded IDs, source MD5s and normalized Flickr author identities
in the pinned complete official validation image metadata. After retaining at
most one image per remaining author and source MD5, 384 eligible train images
remained. The deterministic construction found 27 disjoint quartets (108
images), fewer than the fixed screening cap of 64 quartets. Its first-stage
class coverage excludes Toy, Food, Sculpture and Animal; those shortfalls were
not filled after observing results.

Flickr page checking returned matching license/author declarations for 93/108
images. The other 15 were nine HTTP 404, two HTTP 403, three license mismatches
and one landing-page mismatch. Only **12 of the 27** quartets have all four
pages matching; the remaining 15 groups were not repaired by substituting
images. Those 12 groups contain target counts Door 6, Plant 6, Vehicle 1,
Clothing 8, Furniture 6, Building 10 and Window 11; the other four classes are
absent. The 48 images are a small exploratory candidate inventory, not an
admitted training set or an independent evaluation. A matching page declaration
is not proof of ownership; media availability, visual duplicates, pretraining
overlap and rights are still open. No images or model weights were downloaded,
and no training, validation-label access, or accuracy evaluation occurred.

Private artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-quartet-candidates-20261027-attempt-01/`.
Raw SHA-256: `quartets.json`
`c97478be225c79c9ee14928e2566dd8511a40f84c1375a96c624b78792c33ab5`;
`private-attribution.jsonl`
`c73763d9775a16b840afb4a44fcaec3ece90f0a0fa38fc8763d0fa92ab23ae5b`;
`summary.json`
`f29e8aecaeccbf38c965ca636c8cdf473e4262a1e092f8dc5caee556dced8326`;
`private-page-checks.jsonl`
`a937576c0951de6e006783f07622af26b4bbbfa4a40e007efd68df3a91ebcfb9`;
`page-check-summary.json`
`5b4fc416a45bce5b53419c81aa253b3bcf2a713b8ceb146e8c582be811230cdd`.
The page checker's internal payload hash is over LF text, not raw Windows CRLF
bytes. Inputs, source versions, procedure and code hashes appear in the summary.
