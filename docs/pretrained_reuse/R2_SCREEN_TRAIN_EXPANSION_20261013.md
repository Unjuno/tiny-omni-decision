# Additional training-app annotation audit

Fixed original train-app assignment, next8 apps, first4 lexical annotations each,
at GroundCUA revision5d6845b0116029d46ec762e734701c5b8ce207c3. Selection saved before
fetch;32 pinned annotation files actually acquired and hashed. No replacements.

Apps: Chromium, DuckDuckGo, FontForge, LibreOffice Impress, OnlyOffice Calendar,
OpenBoard, Signal, WordPress.1552 regions,632blank categories (40.72%);920nonblank
before image-bound validity and category checks. These are not yet admitted crops.
The result demonstrates available training diversity but incomplete supervision.
Do not call blank regions negative examples or infer their category from text.

No images, validation/audit contents, feature extraction or training were added.
Existing train/validation/audit membership remains fixed. Before fitting, acquire
only these training images, verify annotation/image pairing and source hashes,
validate bounds/category inventory, and reject exact cross-split pixel/crop
collisions against existing validation. Preserve all held/rejected records and
report coverage. More apps alone do not establish better quality; comparison must
retain an unchanged validation and explicit overfitting safeguards.

Reproduce acquisition: `python -m scripts.audit_screen_train_expansion` (new output
required; existing output is immutable). Output at
C:/CodexArtifacts/pretrained-reuse-r2/groundcua-train-expansion-annotations-01.
Adjacent JSON contains selection/file hashes and per-file category counts.
Ruff passes. Overall screen/Omni quality remains unmet.
