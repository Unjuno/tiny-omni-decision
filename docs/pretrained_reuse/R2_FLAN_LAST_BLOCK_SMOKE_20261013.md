# FLAN final decoder block mechanics

Existing zero-shot FLAN failed condition following. This bounded experiment
checks whether a small adaptation path is executable before any quality run.
It does not replace the failed frozen reference or adopt a new Omni architecture.

Executed `python -m scripts.smoke_flan_last_block` in a separate process:
seed17, first8 existing context-corpus train records, two optimizer updates,
four single-example microbatches/update. AdamW lr2e-5,weight decay.01,clip1;
dropout disabled. Only `decoder.block.7.*` trains:3,147,264 parameters of
76,961,152 total. Encoder, other decoder blocks, embeddings and output matrix
remain frozen and exactly equal to the loaded checkpoint after both updates.

The hash-verified prior loader preserves distinct FLAN output weights. Corpus
SHA256 `a0116e4e2c5024fe39e113461c3b286da3d76b1f8b11940dc195fd5594c2982b`.
Input format is state, condition, letter-labelled unchanged options, and request
for the option letter. Option count varies with the record; no options dropped.
Cross-entropy is over supplied option-letter logits at first decoder position.
Maximum512 input tokens enforced without truncation; no generation training.

Both losses/gradient norms finite: loss1.84460/1.81703, preclip norm7.36395/4.72556.
These are different batches, not a learning curve or demonstrated improvement.
All14 selected parameter tensors changed. Timed update/check loop .682s excluding
load; peak CUDA allocation375,245,824bytes. No cloud compute, validation, sealed
data, saved trained weights, or modification of earlier artifacts.

Result: `C:/CodexArtifacts/pretrained-reuse-r2/flan-last-block-smoke-01/result.json`.
This supports feasibility of a separate fixed-budget training comparison only.
Any full comparison must freeze example order, budgets, validation rule and
transfer tests before it runs. This77M text reference still does not fit the
current VideoPrism+Whisper stack under200M; no product substitution is implied.
