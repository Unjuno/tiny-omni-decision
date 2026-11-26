# Joint readout with training-only option shuffling

Follow-up to the completed joint1024 run, not a new architecture or blind test.
Saved original-validation reversal attribution: choice581/1008->270/1008,
noul675/987->526/987, score225/545->240/545. Identity changes727/583/284.
Binary format alone does not fix this observed fragility. Same displayed-position
counts are not causal proof of a pure position heuristic (binary identity reversal
makes some such counts tautological). No labels removed or redefined.

Single intervention: shuffle candidate order for each training encounter using
an independent Python Random(17). Keep IDs, question, state, target, source,
parent order and all8,192 encounters identical. Record realized permutations;
allow chance unchanged orders. Validation order stays unchanged. Reordering
preserves common state-token budgets because complete candidate token totals
and independent maximum lengths are invariant.

Start from coverage1792 again; do NOT continue joint1024 optimizer/weights.
Same joint marker readout,1,024updates,seed17,AdamW2e-5,wd.01,clip1,FP32,
dropoutoff,10,647,169trainable/22,713,601resident parameters, embeddings/pooler
frozen. Same two-source validation, earliest minimum meanNLL including0,
evaluationevery128, finish budget unless integrity/numeric failure.

Separate output/smoke process; verify exact initial full-validation parity with
previous joint step0, hashes and best reload. Compare fixed1024 and selected
checkpoint to previous joint and independent runs. Repeat only the predeclared
natural/reverse diagnostic after selection; do not use it to select weights,
calibrate, average permutations, retry seeds, or claim fresh held-out evidence.
The diagnostic is now observed development data. No sealed audit/media work.
Candidate remains rejected unless quality and order stability both improve;
no arbitrary extension or product adoption from this bounded experiment.
