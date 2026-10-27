# Same-option train-candidate quartet feasibility

This is an exploratory feasibility calculation on the already observed 257
train candidates, not a new held-out benchmark or training admission. The
existing four-choice task and all historical evaluation sets remain unchanged.
Each image contributes each verified in-scope positive combined with every
three-label subset of its verified in-scope negatives. A quartet shares exactly
the same four labels and fixed order, with a different correct label for each
of its four distinct images. Do not treat unlabeled classes as absent.

Enumerate option sets using SHA-256 order of `quartet-set-v1|17|<sorted IDs>`.
Within each set, process labels in sorted order, and greedily choose the lowest
hash available image using `quartet-image-v1|17|<set>|<label>|<image>`. Mark
all four images globally used, and repeat until the set can no longer form a
quartet; continue to the next set. Fixed option ordering uses SHA-256
`quartet-position-v1|17|<set>|<label>`, identical for all four rows.
Keep all selected and rejected candidate counts, target distribution and
option-set count; never interpret the unselected rows as bad examples.

Under this exact construction any deterministic option-only predictor receiving
only the same ordered options (and no image ID) is correct on precisely one
of each four rows; the expected accuracy of a random option-only predictor is
also 25%. Group identity, labels, images and metadata are not supplied as extra
features to such a predictor. This construction does not prove quality,
guarantee all 11 classes, or provide independent train/evaluation splits.
Do not fit a model or use this subset for checkpoint selection or final quality
claims. Source rights, pretraining overlap, and separate unseen evaluation gates
remain open.
