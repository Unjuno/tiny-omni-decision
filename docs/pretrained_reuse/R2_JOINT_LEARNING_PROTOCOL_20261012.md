# Joint versus independent readout learning

Frozen before training. Two seed17 arms (independent/joint),1024updates each,
8questions/update; same parent order as paired-wording control (no paraphrases).
4096 original encounters plus4096 curriculum encounters;5120 unique parents.
Start from coverage1792 weights in both arms, new AdamW2e-5,wd.01,clip1,
FP32 deterministic, dropout off, all6encoderblocks/classifier trainable,
embeddings and pooler frozen.22,713,601 resident/10,647,169 trainable parameters.
Keep every question and candidate. Use encode_matched to make state-token
prefixes identical between formats, context512;260training/117validation
original questions truncated. Existing full-context scores are separate.

Only readout/tokenization changes: CLS per independent candidate versus joint
contextual separator markers with the same pooler/classifier. It is a composite
readout-format intervention, not isolated attention causality. The initialization
was trained for CLS; initial joint scores need not match. Do not transplant
weights into previously trained image/video heads after changing the encoder.

Same validation IDs/order (2540original+2048curriculum). Evaluate0 and every128;
earliest minimum mean sourceNLL including0 selects best. Keep final separately;
finish planned budget unless integrity/numeric/storage failure. Log all attempts,
exact best reload, frozen/source hashes, time, memory, losses and predictions.
No audit, new seed search, temperature tuning or quality-based early stopping.

Separate two-update smoke each. At completion compare selected and endpoint
probability metrics and Accuracy, state-cluster bootstrap. Then use fixed reverse
option order on selected checkpoints without tuning to measure order sensitivity;
record aligned probabilities and identity changes. All quality is development
validation, not a new blind estimate. No automatic adoption or global95% claim.
