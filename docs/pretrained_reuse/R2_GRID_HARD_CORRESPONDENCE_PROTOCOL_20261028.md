# Original GRID hard correspondence development catalog

Before reading media or predictions, use only the hash-pinned provisional
GRID development split. Within each original speaker and split, index all
six-character codes by their six single-slot wildcard patterns. For each
anchor, select three distinct other codes that differ in exactly one slot,
ordered by SHA-256 of `grid-hard-v1|<anchor>|<candidate>`; skip and count an
anchor if fewer than three exist. Canonicalize and deduplicate each quartet.
Order each quartet's opaque candidate IDs by SHA-256 of
`grid-candidate-v1|<code>`. For every quartet emit all four query roles, each
once as the correct candidate, with identical candidate IDs and order.
The model sees audio content and candidate video content, never codes, speaker
labels, transcript text or target labels. An exactly video-only scorer must
return the same prediction for all four roles of a quartet and can therefore
get exactly one of the four right. Audio-only, media replacement, permutation,
missing-input and timing controls remain mandatory for any later evaluation.

Persist deterministic train and development-validation manifests in new
external storage, plus accounting and hashes. Check same-speaker quartets,
distinct candidates, one-slot distance from the anchor, four balanced targets,
unique canonical quartets, and no source/content/speaker overlap across
splits. All existing Lombard train/calibration/development utterance codes are
already excluded by the source manifests; verify those pinned hashes. No media,
feature, model, validation prediction or sealed-audit data is opened. This is
metadata construction, not admitted media or evidence of actual alignment,
quality or independently verified persons. Speaker identity across corpora and
rights/provenance of each downloaded file remain gates.
