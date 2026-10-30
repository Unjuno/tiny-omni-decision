# Fixed transfer diagnostic after broader text coverage

This is a separate inference-only diagnostic after the completed training run;
it does not change that run's configuration, data or selected checkpoint.
Compare deterministic control best1664 SHA0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882
with coverage best1792 SHAb063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5.
Selection is frozen by the previous validation rule; no further selection.

Use the immutable vocabulary-transfer-inputs-01/inputs.jsonl,
SHA1a825b413e32f319dead47ccbef07ef3b36f2c391ca2c197a694aaeb4ed54dd2.
Ten conditions/10240 questions,128 latent states,1024 base questions.
Retain every arm and target. No new prompt, calibration, training, seed search,
new audit or data filtering. Verify disjoint states from both complete training
sources and verify the existing symbolic targets before inference.

Reuse the existing range_generalization.evaluate implementation with checkpoint
arguments only: FP32 frozen Cross-Encoder, same tokenizer/pair format, batch8,
fail if any input exceeds512tokens. Save all probabilities, Accuracy/NLL/Brier/ECE,
per-task metrics and minimum/maximum paired correctness. Existing paired state
bootstrap10000 draws/seed20261012 for Accuracy. No multiplicity correction or
blind generalization claim. Do not tune from these diagnostic results.

A mean gain cannot compensate for a failed domain-word or numeric-range arm.
Keep old and new artifacts; do not silently swap the shared image/video text tower.
Report regressions and previously unseen-to-this-checkpoint inputs as development
transfer, not lifetime-unobserved evaluation. Product remains four modalities>=95%.
