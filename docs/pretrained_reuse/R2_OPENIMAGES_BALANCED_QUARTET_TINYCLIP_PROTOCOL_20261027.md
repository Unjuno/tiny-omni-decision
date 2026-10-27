# Fixed TinyCLIP diagnostic on the original train quartets

Freeze before inference. Use the 15 previously constructed disjoint quartets
from the 257-image train-candidate inventory, all 60 images and no replacements.
The separate 12 new quartets are not evaluated here because two lack media.
Verify the pinned group file, candidate media manifest, Flickr page receipts,
class scope and each JPEG hash. Each group must have four distinct images,
identical ordered options, and all four distinct labels as targets.

On CPU, run the already audited frozen TinyCLIP checkpoint using the exact
`a photo containing {label}.` prompt, no tuning, on aligned images and a fixed
seed-20261026 derangement over the same 60 images. Use the processor mean image
as a blank control. Reuse per-image scores for the derangement and one blank
image score for all rows. Report Accuracy, NLL, Brier, ECE, per-quartet
correct counts and top-1 changes. An option-only deterministic predictor with
the same ordered option list scores exactly 25% across every complete quartet.

This is a small, post-hoc-selected, train-source diagnostic with four classes
absent, not an independent quality benchmark. The option sets were built from
labels, ownership and pretrained source overlap remain unverified, and no
model selection, fine-tuning, calibration fitting or final evaluation is
permitted. Keep per-row results private and do not modify historical artifacts.
