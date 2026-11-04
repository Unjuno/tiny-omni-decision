# Shared text tower: controlled image-readout protocol

## Question and scope

Can the already resident Decision text backbone replace the separate MiniLM
candidate encoder without degrading the same frozen-DINO image readout?
Independent components exceed 200M; sharing is a possible budget remedy, not
permission to sacrifice quality or redefine general Decision success as CLEVR-4.

Freeze the completed deterministic CONTROL's best step 1664 for this integration
comparison, independent of the still-running numeric condition's outcome.
Checkpoint SHA-256: 0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882.
This is not selection of a final product text model. No active training artifacts
are changed. The zero frozen numeric delta can be omitted when loading this
control onto its identical base; all six trained encoder blocks/classifier must
match the snapshot scope. Realized resident parameter count must be measured.

## Fixed conditions

- Baseline: pinned 17,389,824-parameter MiniLM-L3 candidate embeddings.
- Shared: pinned 22,713,601-parameter Cross-Encoder with the above frozen delta.
- Same 40 candidate prompts and attended-token mean pooling, width 384, no new
  prompt search, label-derived features, normalization or other representation change.
- Same immutable spatial DINO cache, 8424 training images and 512 validation
  images, four taxonomies/ten options, existing split/media checks.
- Same spatial features (CLS plus 16 cells /4), bilinear head 884,992 parameters,
  hidden width128. No image augmentation or encoder updates.
- Both heads start from exactly the same seed17 tensors and use the same sample
  order. AdamW1e-3/wd1e-4, batch64 questions, 24 epochs, finite gradient checks.
- Earliest minimum validation NLL selects best. Keep best/final separately;
  full histories, fixed training-set metrics, per-taxonomy metrics, predictions,
  unique/consumed accounting, timing/VRAM, weight/feature hashes, exact best reload.
- Compare paired validation predictions by image (10000 bootstrap draws, seed17)
  and rotated-image controls. Existing historical results remain descriptive
  references; this new pair is the controlled comparison.

## Execution gates and interpretation

Implement/test on CPU first; a separate small smoke uses a new output directory
only after current numeric GPU training has terminated. Full pair runs only if
smoke verifies finite training, frozen hashes, identical head initialization/order
and exact best reload. Check measured output allowance plus 512MiB disk reserve.
No concurrency with current GPU training, additional seeds, final audit or cloud.

Do not auto-adopt sharing: report Accuracy/NLL/Brier/ECE for each taxonomy and
aggregate, parameter savings in the COMBINED system, and all regressions. The
Cross-Encoder costs more than MiniLM as an isolated image tower; savings only
exist if the SAME loaded text model serves Decision reasoning and media queries.
Matching feature dimensions or a wide uncertainty interval does not prove
noninferiority. This comparison alone cannot establish natural-image reasoning,
video quality, simultaneous fusion, calibrated 95% general performance or release.
