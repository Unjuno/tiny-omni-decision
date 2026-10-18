# Fixed ViLT learning curve

The8-epoch run ends at its best validation NLL with train and validation still
improving. Measure24epochs with the same fixed features, labels, question strings,
readout, loss, seed17, optimizer and sample order. No encoder changes or new data.

Start fresh, not a partial optimizer resume. Load exact hashed feature arrays,
reference config and history. Require complete equality of the first8 history
records before epoch9; fail closed if they differ. Stop at24 without automatic
extension. Preserve best by earliest minimum validation NLL, final separately,
and require exact best reload. Save full predictions and rotated-image control.
Compare against original8-epoch ViLT and DINO87.40% using the same512 image IDs,
paired bootstrap10000/seed17. This is reused development validation, not blind.

Small separate-process smoke verifies cache reload and preserves the original
smoke's first-epoch outputs. No old artifact is overwritten or re-extracted.

Metadata erratum: original config `representation="CLS384 + mean-patch384"`
was copied from the DINO scaffold. Executed code used ViLT `pooler_output`,
and saved arrays are `[N,4,768]`. Keep the original config unchanged as evidence;
the new config names the actual representation. This does not alter the prior
predictions or metrics. New `extraction_seconds` measures cache load, not encoding.
