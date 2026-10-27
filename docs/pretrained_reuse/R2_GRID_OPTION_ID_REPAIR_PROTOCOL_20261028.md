# GRID quartet option-ID repair

The original two-slot catalog encoded each six-character utterance using an
unsalted public hash as the option ID. The code space is small enough that an
ID can be reversed with a dictionary. Although the native model backends do
not receive text labels, these identifiers are public runtime fields and must
not encode transcripts. Before any GRID model evaluation, create a separate
catalog from the hash-pinned original: retain every quartet, speaker, ordered
code, query role and source split exactly; replace the four IDs with local
`option-0` through `option-3` and remap each private target by its original
candidate position. Check one-to-one mapping, four balanced targets, row
counts, original-to-new order and output hashes. The quartet ID and raw codes
remain private dataset metadata and never enter the model input. No media,
model output, training or sealed audit is read. Preserve the original catalog.

This fixes a label-exposure path in public identifiers, not all possible
shortcuts. The actual adapter must still pass only media and per-request IDs;
all missing/swap controls, source and speaker isolation checks remain required.
