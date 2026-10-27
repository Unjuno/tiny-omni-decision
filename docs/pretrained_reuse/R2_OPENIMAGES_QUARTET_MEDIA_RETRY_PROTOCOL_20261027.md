# Retry two unavailable quartet images

Freeze before requests. Take only the two failed rows from the fixed 48-image
acquisition receipt, and verify its raw SHA-256 and the pinned page receipts.
For the HTTP 429 original-image request, wait at least five minutes after the
first attempt completed, then retry the exact same URL once; stop on another
429 without further requests. For the HTTP 410 thumbnail, try once the same
photo's original URL already recorded in its source metadata, only if its
HTTPS Flickr CDN host and numeric photo ID match the checked landing page.
This is the original image of the same photo, not a replacement question.

Use the existing JPEG download cap and a new external directory. Maintain at
least 1 GiB free plus 8 MiB before each request. For either saved original,
require its raw MD5 to equal the original MD5 declared in the pinned source
metadata; retain any mismatch only as a rejected diagnostic artifact, never
as admitted media. Store SHA-256 and duplicate warnings against existing train
candidate, quartet and validation image caches. Preserve all prior artifacts.
No validation labels, model training, candidate replacement or quality claim
is permitted. A checked page and source MD5 do not establish ownership.
