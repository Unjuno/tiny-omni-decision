# Same-option quartet feasibility on the full train metadata cohort

Before the full source scan, fix the following exploratory procedure. Join
the previously pinned official train image metadata IDs with the verified
labels in the pinned full annotation object, checking each full object's
generation, ETag, length and SHA-256. Restrict to the frozen 11-class scope,
explicit positive labels and at least three distinct explicit negatives.
Apply the same fixed deterministic greedy quartet procedure as in the prior
257-image feasibility diagnostic to every qualifying train image; do not use
image media, predictions, validation/test labels or any evaluation scores.

Keep only aggregate counts: eligible images, possible and complete option
sets, greedy disjoint quartets, unused images, and selected target counts per
class. Image IDs and label-level records remain transient in memory and are not
saved. The greedy result is not an optimal matching, a sampling plan, an
admissible dataset, or a performance estimate. Metadata license declarations
do not authenticate individual rights. Actual new candidate acquisition
requires a separately frozen provenance, author and split exclusion rule and
per-image rights and media checks. Do not modify or filter existing evaluation.
