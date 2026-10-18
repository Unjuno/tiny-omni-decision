# Corrective matched readout follow-up (2026-10-19)

The first 100k readout used 64 images per update (256 taxonomy decisions) and
gradient clipping, unlike the older reference runner's 64 decisions per
update without clipping. Preserve that completed run as exploratory; do not
describe it as an optimizer-matched data-volume comparison.

This corrective run uses the same frozen features, labels, seed 17, scorer,
option embeddings, AdamW (0.001 / 0.0001), and eight epochs. It fixes both
implementation differences by using the prior runner's unit of one
image-taxonomy question per item, 64 questions per optimizer update, and no
gradient clipping. Taxonomy questions are ordered image-major before
the fixed seed-17 shuffle, matching the previous runner's task construction.
Validation and the earliest minimum macro validation NLL selector remain
unchanged. The full epoch budget runs and both best and final checkpoints are
retained.

This follow-up was specified after the validation had already been observed
for the first run. The validation is therefore development data with prior
exposure; the follow-up does not create a blind or sealed result. Report both
runs and the older reference without cherry-picking. If their conclusions
differ, characterize the 100k data-volume effect as uncertain.
