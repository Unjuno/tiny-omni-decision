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

## Completed result

Execution PASS:2,048updates, all17scheduled validations, exact best-logit reload,
unchanged frozen hash, source hashes preserved. Best selected2048; best/final
saved separately with identical tensors. Run-wide and selected consumption both
16,384unique questions,8,192per source. Underlying latent curriculum states remain
1,024; this is range augmentation, not additional independent states.

Training source9c66aec; transfer runner4a773d3. Same initialization, sample order,
validation IDs/order, optimizer/budget as encoder-coverage. Only curriculum
training numbers/thresholds changed. Transfer runner checks completed history,
predefined selector, exact checkpoint hashes and unchanged source configuration
before reading model predictions. Neither training nor selection loads transfer.

### Original fixed validation curve

| Step | Original Accuracy | Original NLL | Curriculum Accuracy | Curriculum NLL |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.58110 | 0.93779 | 0.23389 | 1.45684 |
| 128 | 0.57047 | 0.93655 | 0.23828 | 1.45957 |
| 256 | 0.57323 | 0.94645 | 0.24170 | 1.44258 |
| 384 | 0.57874 | 0.93855 | 0.24854 | 1.44147 |
| 512 | 0.57008 | 0.93723 | 0.24854 | 1.44392 |
| 640 | 0.57598 | 0.93478 | 0.24951 | 1.44563 |
| 768 | 0.58110 | 0.91959 | 0.23779 | 1.44100 |
| 896 | 0.58976 | 0.90259 | 0.23145 | 1.44190 |
| 1024 | 0.59055 | 0.91298 | 0.24219 | 1.44027 |
| 1152 | 0.59094 | 0.90465 | 0.23242 | 1.43937 |
| 1280 | 0.59961 | 0.90742 | 0.23877 | 1.43826 |
| 1408 | 0.60394 | 0.89305 | 0.23828 | 1.44085 |
| 1536 | 0.59843 | 0.90826 | 0.23047 | 1.44005 |
| 1664 | 0.60157 | 0.87891 | 0.22803 | 1.43997 |
| 1792 | 0.60472 | 0.90207 | 0.22559 | 1.43743 |
| 1920 | 0.61063 | 0.88739 | 0.23877 | 1.43163 |
| 2048 | 0.61024 | 0.88105 | 0.24512 | 1.43305 |

Selected/final original Accuracy61.02%,NLL0.88105,Brier0.49768,ECE0.03485. Prior encoder-coverage selected original Accuracy60.24%.
Curriculum Accuracy24.51% versus prior80.57%. The narrow-format learning gain was
not retained under this wider training distribution at the same budget.

### Preregistered post-selection transfer

| Model | Accuracy | NLL | Brier | ECE15 |
| --- | ---: | ---: | ---: | ---: |
| prior encoder coverage | 0.30664 | 2.39435 | 0.95979 | 0.33344 |
| numeric-range candidate | 0.24268 | 1.44305 | 0.76231 | 0.01086 |

| Numeric transform | Prior Accuracy | Candidate Accuracy |
| --- | ---: | ---: |
| 3*x+11 | 0.28711 | 0.25977 |
| 5*x+23 | 0.25391 | 0.25195 |
| 12*x+11 | 0.33789 | 0.21094 |
| 16*x+23 | 0.34766 | 0.24805 |

Candidate-minus-prior Accuracy:-6.40points, paired state95% percentile interval
[-8.98,-3.86]points,256states/8questions per state,10,000draws,seed20261012.
This is a predeclared development transfer comparison, not the sealed audit.
The last two transforms extend beyond the augmented range; first two are largely
within it but use unseen coefficients. Candidate accuracy remains near random
across all four groups. All per-task metrics and predictions are retained.

## Interpretation

Quality improvement NOT SUPPORTED within this fixed budget. Calibration/loss
improve mainly by removing prior overconfidence while accuracy falls; this is
not useful generalized numeric reasoning. Uniform-reference curriculum Accuracy
is23.75%,NLL1.44208,Brier0.7625. The candidate transfer metrics are near those
values. Do not promote it or select a different checkpoint using transfer scores.

This does not isolate whether more optimization, better numeric representation,
or greater independent data diversity would help. The corpus still has only
1,024latent training states and single-pass question coverage. Changing numeric
coverage also changes task difficulty at a fixed budget. We cannot infer an
absolute capacity ceiling or impossibility. Rolling mixed training CE cannot
supply a fixed-set generalization gap. Before another training run, measure the
selected model's fit on a fixed training slice to distinguish failure to fit from
poor transfer; no additional seed or budget has been started.

Text/image/audio/video95% remains unmet. No model or benchmark simplification,
quantization, deployment promotion, cloud cost or sealed evaluation occurred.

## Artifacts / compute

Local FP32 training/evaluation/reload 1119.285s (load/tokenization excluded); transfer 22.294s. Peak training allocation 3,473,190,400bytes; transfer 194,127,360bytes; max transfer input 162tokens, no truncation.
Total model22,713,601parameters; trainable10,647,169; no added parameters.
Weights-only artifacts are not exact resume checkpoints. Hashes/config/history/
all predictions stay in new output paths with compact evidence in adjacent JSON.
Ruff passed; CPU tests333passed,1skipped before execution of transfer comparison.
