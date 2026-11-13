# Training-app image admission

All32 preselected training screenshots were actually acquired from the pinned
revision, matched to annotation paths and hashed. From1552 source regions:
906 valid labeled regions admitted,632 blank and14 invalid boxes held with
explicit IDs/reasons. No clipping, label inference or example replacements.
906 admitted crops are byte-unique after decoded-pixel hashing.

Exact decoded-screen and admitted-crop overlap with the existing validation is0.
This does not rule out near-duplicates or shared visual styles. Validation/app
membership remains unchanged; audit images/annotations remain unopened.
The original32 training screens plus these32 provide16 training apps, but no
expanded training has run yet. Preserve all existing small-pilot checkpoints.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r2/groundcua-train-expansion-images-01.
records.json pins image/annotation hashes and sizes; admitted.json pins crop
bounds/hashes; held.json lists exclusions. All raw images remain outside Git.
Adjacent JSON includes file hashes. Reproduce with
`python -m scripts.prepare_screen_expansion_images` in a new output directory.
Ruff passed. This completes input admission, not feature extraction or quality.
