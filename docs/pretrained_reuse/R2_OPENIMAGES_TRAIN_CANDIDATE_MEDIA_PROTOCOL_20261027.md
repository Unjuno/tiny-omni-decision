# Open Images candidate media acquisition and duplicate screen

Freeze before downloading. Use only the 258 candidates whose recorded Flickr
page license and author declarations matched, from the pinned candidate and
page-receipt artifacts. Keep the other 39 excluded. Verify the HTTPS CDN host
and that the CDN photo ID matches the checked landing page. Use existing JPEG
download, decode, normalized RGB hash and perceptual hash code, with its 8 MiB
per-image limit. Write to a new external output directory; never overwrite.

Use sequential requests and a cumulative 128 MiB budget. Before each request
require 1 GiB free plus the full 8 MiB maximum possible in-flight image; stop
and record an incomplete run if this or the cumulative budget fails. Check
candidate normalized hash and 64-bit perceptual hash against both previously
saved validation media manifests; report exact hashes and hamming distance <=4
as duplicate warnings, not definitive visual identity. No validation labels,
model weights, training or official test data are accessed. A matching Flickr
page is not independent ownership verification; media is not admitted until
remaining rights, option-role, duplicate and independent-evaluation gates pass.
