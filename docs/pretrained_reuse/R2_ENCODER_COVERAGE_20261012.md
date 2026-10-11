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
