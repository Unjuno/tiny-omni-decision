# Predeclared numeric-range coverage experiment

Controlled change: replace only curriculum training values/thresholds by positive
affine transforms, keeping question templates, labels, schemas, cardinality,
row/sample IDs, option order, targets and number of underlying states fixed.
Four balanced state-level transforms: x,2*x+7,4*x+31,8*x. Each receives256
states/2,048questions. Same state is not duplicated. These are transformed
versions of the existing1,024training states, not8,192new independent assets.

Keep all-six-encoder-block training, frozen embeddings/pooler, preserved original
epoch3 initialization, new AdamW, seed17,2,048updates,4+4batches,2e-5constant LR,
.01weight decay, clipping1, dropout disabled, original data and validation exactly
unchanged. Same sample IDs/order. Same earliest mean(original,curriculum) NLL
selector including step0, every128steps. Do not resume from the successful
encoder-coverage weights; that would confound the training-data comparison.
Save best/final separately, check hashes and exact best reload. One local run,
no automatic budget extension, extra seed or paid compute. Separate2-step smoke.

After checkpoint selection is frozen, compare the original encoder-coverage
selected weights and the new selected weights once on2,048questions/256new
seed59states. Fixed transforms3*x+11,5*x+23,12*x+11,16*x+23,64states each.
First two test unseen affine transformations mostly within the augmented range;
last two extend beyond it. Report separately, not only pooled. Transformation
coefficients being unseen is not proof every numeric value is unseen.
Old train/validation and seed43diagnostic state groups are excluded. No negative
numbers/decimals, cardinality or new paraphrases are tested here. No95%claim can
follow from this scope. The earlier observed diagnostic is not rerun or used for
selection. This new set is a development transfer test, not the product sealed
audit. Do not tune or retry a seed after viewing its result.

Record curves, per-task Accuracy/NLL/Brier/ECE, selected/final values, unique
counts, bytes, compute and same-condition original-task retention. Accuracy may
fail to improve; preserve that result. A successful range change would not repair
the independently observed wording weakness or prove multimodal reasoning.

Outputs: C:/CodexArtifacts/pretrained-reuse-r2/numeric-range-curriculum-01,
conditional-repair-numeric-smoke-01 and conditional-repair-numeric-range-01.
Corpus hashes and transform counts are in adjacent JSON. All old artifacts stay
unchanged. Local estimate20minutes, available disk checked before starting.
