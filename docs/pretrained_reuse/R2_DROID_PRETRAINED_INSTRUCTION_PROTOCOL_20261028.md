# DROID frozen pretrained-instruction probe

Before execution, retain the same 275 records, past-index text, 268
instruction-connected components, seed-17 folds, 200 full-batch Adam updates,
learning rate, regularization, metrics and CPU execution as the
[word-hash probe](R2_DROID_INSTRUCTION_PROBE_PROTOCOL_20261028.md).
Replace only its hashed-word input by the 384-dimensional final CLS hidden
state of the locally pinned Apache-2.0 `cross-encoder/ms-marco-MiniLM-L6-v2`
checkpoint, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Empty text
is the zero vector with a separate missing indicator. Encoder is frozen in eval
mode, batches use the local tokenizer with truncation at 128 tokens. Check all
model and tokenizer file hashes against the previous R1 receipt before use.

Retrain both sensor-only and sensor-plus-embedding heads on identical folds;
save raw out-of-fold probabilities and all metrics. No validation or sealed
audit score is used. This tests representation under an already examined train
cohort, not a held-out quality claim. A retrieval-trained CLS representation
may not be appropriate for control intent; its 22.7M parameters must be counted
alongside every other deployed encoder before evaluating the 200M constraint.
Inspect tokenizer truncation and missingness; do not assert overall model quality
or select a final product from this probe.
