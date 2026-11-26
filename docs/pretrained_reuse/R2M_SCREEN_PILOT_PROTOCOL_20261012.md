# Screen category pilot: frozen preparation

GroundCUA revision5d6845b0116029d46ec762e734701c5b8ce207c3, MIT declaration.
Group all87 apps before opening new annotations. Inspected7-Zip/Anki/Audacity
belong to train; sorted remaining84 apps shuffled with Python seed17 produce
57 additional train,10validation,17sealed-audit apps. Audit contents stay unopened.

Bounded pilot: first8 train and first4 validation apps, first4 lexical screenshot
filenames each. Freeze48 identities before downloading. No category-based screen
replacement. Record all empty labels, invalid bounding boxes, app/category counts,
raw-file hashes and decoded pixel hashes. Reject cross-split identical screenshots.
Near-duplicate UI components across apps remain a limitation to quantify.

Next classifier probe uses only pixels from supplied regions, frozen DINOv2
CLS+mean patches, no annotation text or filename features. This tests coarse UI
perception with oracle localization; it does not measure autonomous action choice.
Class inventory and missing-label policy must be fixed before fitting, with all
source category coverage disclosed. Unlabeled regions cannot supply supervised
targets; do not silently turn them into a class or imply whole-screen coverage.
No classifier is trained by this preparation script. Fusion remains a separate
unmet task requiring complementary paired observations.
