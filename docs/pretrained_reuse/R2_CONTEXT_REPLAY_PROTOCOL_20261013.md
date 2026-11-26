# Fixed original-task replay comparison

The all-encoder context adaptation improved context decisions but degraded original
task retention, including untruncated questions. Test a fixed replay allocation
before any further increase in model size, learning budget or compression.

Preflight created a real ordered3072-question file:1536context +1536original train
examples, no repeated IDs. Each batch has4context then4original. Context order is
the first1536 of the prior fixed treatment order; original order uses seed17.
Realized context kinds:527necessary,522irrelevant,487plain. No rebalancing or drops.
Checked zero ID/canonical-state/group overlap with both existing development sets.
Actual tokenizer max879tokens;213candidate pairs need the historical state-only
512token truncation. Keep and report those rows; do not silently exclude them.

Next run: fresh optimizer from the same coverage parent, all6encoder blocks plus
classifier,384updates,batch8,AdamW2e-5/.01,clip1,dropout disabled,seed17. Embeddings
and pooler frozen. Separate output directory and separate-process two-update smoke.
Fixed final endpoint only; context and original validation at0/128/256/384. Save
all predictions, training CE/gradient norms, hashes, frozen checks and exact reload.
Existing context-only final384 is the comparator; do not rerun or replace it.

This tests budget allocation to rehearsal: context exposure is halved. It does not
isolate adding original examples at fixed context exposure. No loss-weight sweep,
epoch selection or additional budget. Report all task/condition metrics and costs;
neither restored aggregate Accuracy nor context-only gains alone qualify adoption.

Preflight artifacts: C:/CodexArtifacts/pretrained-reuse-r2/context-replay-preflight-01.
Order hash86dd9a7e003267926c80f317a7ddbddef534a0ed8c6b15e42239774a475e5e09.
Adjacent JSON pins source hashes. Sampler tests2passed; Ruff passed. This commit
does not claim training started. Existing corpus/checkpoints remain unchanged.
