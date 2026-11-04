# Predeclared encoder-coverage comparison

The completed 2,048-update last-block experiment did not establish useful
conditional reasoning. Test one factor: train all six existing BERT encoder
blocks and the classifier, instead of the final block/classifier. Embeddings
and pooler remain frozen. Total parameters stay 22,713,601; expected trainable
count 10,647,169 instead of 1,774,849. This is selective adaptation of an existing
small encoder, not training a new large model.

Keep the previous epoch3 initialization (not the failed repair weights), new
AdamW, seed17, constant2e-5 LR, dropout off, batch4+4, 2,048 updates, no repeats,
identical sample order, original/curriculum data and validation IDs, 128-step
evaluation, earliest minimum mean-NLL selector including step0, clipping1,
weight decay.01 and tokenization. No other factors change. Save best/final
separately, verify exact best reload and frozen embeddings/pooler hashes.

First execute a separate two-update smoke to measure VRAM/time and verify the
trainable count, initial logits and frozen hashes. If safe, one bounded full run
uses a new output directory. No paid compute, new seed, sealed access or automatic
budget extension. Local estimate roughly 20–30 minutes; actual timing recorded.
Compare the same fixed step512/1024/1536/2048 and selected checkpoint against the
last-block run. Record original-task retention and all curriculum task metrics.
Any advantage remains single-seed synthetic development evidence, not the four
modality95% gate. Failure does not prove a universal capacity ceiling. If accuracy
remains near chance despite full unique coverage, do not keep repeating this
setup; reassess representation and supervision before another training run.

Implementation preserves the legacy default last_block policy and validates the
initial delta against that original parameter set before enabling broader
training. Tests verify six-block coverage, frozen embeddings/pooler and legacy
policy equivalence. New output root:
C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-encoder-coverage-01.

## Completed run

Execution PASS. Full 2,048 updates and all 17 validations completed; best selected
at 2,048 by the frozen mean-NLL rule. Exact best-logit reload passed. Best/final
files saved separately, with identical tensors and different container hashes.
All pinned source hashes unchanged; frozen embedding/pooler hash unchanged.
16,384 distinct questions consumed: 8,192 original and 8,192 curriculum.
Here selected and run-wide consumption match because the selected step is final.
Total parameters 22,713,601; trainable 10,647,169 (98 tensors). No new parameters.

Source5ff2650, comparison5106557. Command: `python -m scripts.train_conditional_repair --root C:/CodexArtifacts/pretrained-reuse-r2 --model C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model --output C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-encoder-coverage-01 --trainable-policy all_encoder_blocks --steps 2048`.

### Learning curve

| Step | Original Accuracy | Original NLL | Curriculum Accuracy | Curriculum NLL |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.58110 | 0.93779 | 0.23389 | 1.45684 |
| 128 | 0.57992 | 0.93342 | 0.24219 | 1.45776 |
| 256 | 0.57402 | 0.94553 | 0.24463 | 1.44217 |
| 384 | 0.57677 | 0.93961 | 0.24707 | 1.44112 |
| 512 | 0.56811 | 0.93727 | 0.25195 | 1.44303 |
| 640 | 0.57835 | 0.93128 | 0.24561 | 1.44549 |
| 768 | 0.58307 | 0.91666 | 0.24219 | 1.43941 |
| 896 | 0.58189 | 0.90139 | 0.22852 | 1.44094 |
| 1024 | 0.58937 | 0.91320 | 0.26465 | 1.43289 |
| 1152 | 0.58898 | 0.91543 | 0.35400 | 1.33287 |
| 1280 | 0.59449 | 0.89345 | 0.65771 | 0.80038 |
| 1408 | 0.60551 | 0.88631 | 0.71436 | 0.65965 |
| 1536 | 0.59843 | 0.88148 | 0.75586 | 0.59256 |
| 1664 | 0.60709 | 0.87768 | 0.78027 | 0.52493 |
| 1792 | 0.59921 | 0.88704 | 0.79736 | 0.49212 |
| 1920 | 0.61299 | 0.87863 | 0.80127 | 0.46608 |
| 2048 | 0.60236 | 0.87743 | 0.80566 | 0.46110 |

### Final / selected metrics

| Corpus | Accuracy | NLL | Brier | ECE15 |
| --- | ---: | ---: | ---: | ---: |
| original | 0.60236 | 0.87743 | 0.49486 | 0.01919 |
| curriculum | 0.80566 | 0.46110 | 0.25546 | 0.03056 |

| Task | Count | Accuracy | NLL |
| --- | ---: | ---: | ---: |
| original:choice | 1008 | 0.57738 | 1.02235 |
| original:noul | 987 | 0.70213 | 0.54844 |
| original:score | 545 | 0.46789 | 1.20521 |
| curriculum:conditional_selection | 256 | 0.58984 | 1.08260 |
| curriculum:count | 256 | 0.41016 | 1.19468 |
| curriculum:extremum | 1024 | 0.89355 | 0.25232 |
| curriculum:temporal_order | 512 | 0.93555 | 0.20111 |

### Controlled comparison and uncertainty

Same initialization, complete training order, validation IDs/order, source hashes,
optimizer settings and budget. Initial metrics exactly match. Only training
coverage within the existing encoder changes. At2048, curriculum Accuracy rises
from23.14% (last block) to80.57% (all six blocks), a57.42-point difference.
Paired state bootstrap95% interval [55.27,59.57] points,256 independent generated
states, eight questions per state,10,000 resamples, seed20261012, percentile method.
The interval is descriptive: this development validation also selects checkpoints;
it is not a blind product audit, and multiple step comparisons are not adjusted.
At512 the difference was only0.83points with interval[-0.88,2.54]; the late
learning transition is why the planned full budget mattered.

Original Accuracy at2048 is60.24% versus57.28% in the matched last-block run.
Original NLL/Brier/ECE also improve. Original Accuracy peaks61.30% at1920 then
falls at2048, while meanNLL selects2048. Keep that trade-off visible; do not select
1920 post hoc for a preferred metric. No fixed training-set generalization gap was
measured; rolling training CE is over changing samples. The curve is not proof
that overfitting is absent. Additional seeds have not been run.

This supports broader encoder adaptation for this synthetic conditional task.
It does not establish general numerical/language reasoning or four-modality
quality. Count41.02% and conditional-selection58.98% remain weak. Extremum89.36%
and scalar-time-order93.55% are stronger, but scalar time ordering is not video
reasoning. No95% claim, model promotion, quantization or sealed audit followed.

Next evidence to collect: whether the learned behavior transfers to the preserved
known-failure condition probe with different candidate labels, numeric range,
field names and state schema, using the selected weights once with unchanged
input rendering. That is an observed development diagnostic, not a new blind test.
Do not extend this run automatically or infer a universal model-capacity ceiling.

### Resources / artifacts

Local RTX3080 Laptop16GB, FP32, Python3.11.9, torch2.6.0+cu124,
Transformers5.6.2. Timed training/evaluation/reload1,140.98s versus884.70s
for last-block; approximately29% longer. Tokenization/loading precede the timer.
Peak PyTorch allocation3,473,190,400bytes versus802,154,496bytes; whole-device
usage also includes reserved memory and other processes. No cloud cost.
Weights-only checkpoints are not exact resume artifacts. Pinned configs, all
predictions, all history and artifact hashes are recorded in adjacent JSON.
No legacy or original reference weights/data were changed. Sealed audit unopened.

CPU tests326 passed,1 skipped; Ruff passed. Comparative analysis has explicit
ID/options/target/probability integrity checks and self-comparison returns zero.
