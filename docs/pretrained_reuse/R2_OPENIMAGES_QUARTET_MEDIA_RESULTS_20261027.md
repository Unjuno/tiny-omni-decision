# Train quartet candidate media acquisition

The [fixed procedure](R2_OPENIMAGES_QUARTET_MEDIA_PROTOCOL_20261027.md)
attempted all 48 images in the 12 preselected page-matched quartets. It saved
46 JPEGs, totaling 4,217,389 bytes. Two requests failed: one HTTP 429 and one
HTTP 410. Ten quartets have all four images; two are incomplete. No images
were substituted and this does not establish the planned 12-quartet comparison.

There were zero exact normalized RGB or perceptual hash distance <=4 pairs
against the previously saved 257 train candidates or the two saved validation
media caches, and zero such pairs within the newly saved images. These hashes
do not detect every visual duplicate, crop, near scene, or unseen validation
asset. Page declarations are not independent ownership verification; rights,
pretraining overlap and independent evaluation remain open. No model, model
training, validation labels, or test labels were used in this acquisition.

External private output:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-quartet-media-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`aedf53e3f2a30c6602b08843e688d720947df85be6807ee743cda83329d7d8a5`;
`private-media-manifest.jsonl`:
`1ff2979c09722bd8554f6ac832d578d3483db1de2862a89d4f8958046a5c6c02`;
`private-duplicate-warnings.json`:
`3db8428aef588382c93ca26eb789d41816612734aa08543fa4a4b5e44f4643f1`.
