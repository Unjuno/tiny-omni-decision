# Frozen image candidate-interaction control

Current concat-MLP DINO readout has poor counting despite strong color/shape.
Test one alternative readout, not a new backbone: two linear projections to128
dimensions, image and question/candidate, with dot product divided by sqrt128.
147,712 parameters versus147,713 for concat1152→128→1/GELU. No normalization,
new task head, option-ID feature, target-derived prompt or encoder update.

Reuse exact CLS384+patch-mean384 caches from dinov2-clevr4-probe-01 after SHA256
checks. Same8,424 train images and512 selection-validation images, four questions
each. New output for each policy, fresh seed17,8epochs,AdamW1e-3/wd1e-4,batch64,
same shuffled question IDs. Select earliest minimum validation NLL. Best/final
separate, reload and frozen-file hashes verified. Smoke separate process first.

Rerun concat control in the same cached pipeline. Require its eight-epoch history
to equal the original before interpreting a readout comparison. Report all
four taxonomy metrics, train/validation curve, unique/consumed examples, rotated
image control, bytes/parameters,time/VRAM. Paired512-image bootstrap10,000 draws,
seed17,percentile95% of selected Accuracy difference; development selection bias
remains. No final reporting subset/sealed audit and no budget extension.

This changes interaction parameterization and initialization, not just whether
text is present. A gain supports a better readout under this budget; failure
does not prove the frozen representation lacks count information. Retain the
best measured existing reference, without claiming general image95%.
