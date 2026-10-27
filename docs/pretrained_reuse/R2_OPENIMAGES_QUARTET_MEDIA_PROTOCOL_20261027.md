# Train quartet candidate media acquisition

Before acquisition, pin the existing 27-group candidate file and page-check
receipts. Attempt exactly the 48 images belonging to the 12 groups with all
four page declarations matched. Do not substitute unavailable images. Require
the existing Flickr HTTPS host and photo-ID checks, bounded JPEG download and
decode, per-image SHA-256 and perceptual hash. Write only to a new external
directory; no source artifacts may be overwritten. Use sequential downloads,
an 8 MiB individual and 64 MiB cumulative limit, and require at least 1 GiB free
plus 8 MiB before each request. Record incomplete runs and exact failures.

Compare decoded RGB hashes and 64-bit perceptual hash distance <=4 against the
previous 257 train candidate images and the two saved validation media caches,
and among new images. Keep exact and approximate matches as warnings; do not
silently remove examples. Report complete groups after acquisition. No model,
training, validation labels, final-test data, or candidate substitution is
allowed. Page matching is not independent ownership verification. These are
train-source development candidates, not an independently validated benchmark.
