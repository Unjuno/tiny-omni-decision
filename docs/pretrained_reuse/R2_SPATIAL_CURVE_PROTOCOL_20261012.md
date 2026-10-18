# Fixed 24-epoch spatial/global learning curve

The matched884,993-parameter heads were still improving at epoch8; neither
condition established a clean plateau. Before interpreting the absence of a
spatial gain, give both the same24-epoch budget. No encoder or text adaptation.

Fresh seed17 initialization, same head6912→128→1, AdamW1e-3/wd1e-4, batch64,
sample shuffle order, input normalization and train/selection-validation IDs.
Train8,424 images × four taxonomies; validation512 images × four taxonomies.
Both conditions use identical initialization and ordering. No new source,
augmentation, label, sampler, feature shape or learning-rate schedule change.
Fixed24epochs, earliest minimum validation NLL selects best; final stays separate.

Reuse byte-hashed train/val cells from dinov2-spatial-control-02. Verify parent
config and eight-epoch histories, original corpus manifests, base model hashes,
cache shape/finiteness and global-feature parity before training. Do not overwrite
parent caches or checkpoints. Separate smoke process with eight images/split and
one epoch tests execution. Full run starts fresh, never from smoke/parent weights.
At epoch8 require exact prior history equality before proceeding; no relaxed
tolerance or fabricated exact resume. Fresh run is not optimizer resume.

Record all train/validation curves and taxonomies, selected/final weights,
selected/full consumption separately, unique images/questions, gradient norms,
same-order hash, VRAM/time, reload parity and rotated-image control. Report8/16/24
and selected epochs. Pair spatial-minus-global across512 images with10,000-draw
seed17 percentile bootstrap; validation-selected outcomes are development results.
Previous1595-image reporting subset and sealed audit remain unopened.

No quality-based stopping before24; stop on integrity/nonfinite failures. No
automatic extension after24, no image95% claim from selected taxonomy alone.
This isolates training budget at fixed architecture and tests whether spatial
information helps with more optimization; no capacity-only or backbone-limit claim.
