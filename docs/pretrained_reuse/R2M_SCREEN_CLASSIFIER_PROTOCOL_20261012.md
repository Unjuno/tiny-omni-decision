# Frozen-feature screen region classification

Use all valid nonblank category annotations in the fixed48screen pilot. The8
publisher categories, in fixed order: Button, Menu, Input Elements, Navigation,
Sidebar, Visual Elements, Information Display, Others. Unknown nonblank labels
fail closed; blanks are counted as unlabelled, not guessed. Invalid boxes are
reported, not clipped into invented annotations. All source regions remain in
coverage denominators. No data-dependent image replacements.

Input: annotated region pixels only (oracle localization), outward integer crop,
224x224 bicubic; frozen pinned DINOv2 CLS+patch mean768features. No annotation
text, filenames, app IDs, bbox coordinates or category names as model features.
Train-only mean/std normalization, Linear768->8. Seed17 AdamW lr0.001 wd0.01,
batch64,30epochs, earliest minimum validation NLL includingepoch0. Save selected
and final separately. Compare training-label frequency prior on the identical
validation rows. Report Accuracy, balanced Accuracy, NLL, Brier, ECE, per-class
supports and per-app metrics. No extra seeds or hyperparameter search. Confirm
exact selected reload and retain all epoch predictions and loss curves.

Exact decoded crop duplicates across splits must fail before fitting. Within-split
duplicates remain counted; report unique crop counts. Audit apps remain unopened.
This is a small development pilot on4heldout apps, not blind action selection,
whole-screen understanding or a complete Omni quality test. A separate tiny
training-only classification smoke must pass before fitting the full pilot.
