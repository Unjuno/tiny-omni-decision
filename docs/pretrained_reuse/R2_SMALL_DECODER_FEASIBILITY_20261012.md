# Small instruction decoder adaptation feasibility

The fixed zero-shot SmolLM2 reference failed conditional decisions. Before a
learning-curve experiment, verify trainability/cost on independent development
training data. This smoke is not quality evidence or model selection.

Use pinned SmolLM2-135M-Instruct revision12fd25f77366fa6b3b4b768ec3050bf629380bac,
weights SHA5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c.
No inherited adapted weights. Enable only model.layers.26–29 and final
model.norm; embeddings, LM head and first26layers remain frozen. Tied embeddings
remain counted once. Do not full-finetune, add heads, quantize or train media.

Use identical train ID order and validation IDs as the completed comparison-word
curriculum run: original Typed Decisions plus generated comparison-vocabulary
curriculum. Verify all four source hashes and split states. Remap presented
options in their existing order to A..Z labels; retain full question/state/options.
Use publisher chat template, next-token option logits, restricted softmax/loss.
No target/oracle/split metadata in prompts. Do not truncate, discard examples,
reduce option count, or use observed transfer probes for this smoke.

Preflight all prompt lengths and label tokenization before model execution.
FP32/SDPA,seed17,model.eval to disable dropout,AdamW2e-5/wd.01,clip1. Use only
first two scheduled8-example training batches for two optimizer steps in a
separate process/output. Save effective config, hashes, exact trainable paths,
frozen tensor hashes before/after, norms, timings/VRAM and delta. On the same
training batch verify final-token-only projection agrees with full LM projection;
save/reload the delta and require output parity. No validation quality inspection.
Measure an additional no-grad batch of the eight longest scheduled train prompts
to bound evaluation memory; do not update on it. Never silently shorten sequences.

134,515,008 language +22,056,576 DINOv2 +37,760,640 Whisper =194,332,224
parameters before bridges/readouts, leaving5,667,776 under200M. This is only
arithmetic feasibility, not a quality-tested Omni architecture or a video claim.
Keep previous encoders/readouts/artifacts. Use measured throughput/lengths to
freeze a later learning-curve budget; do not interpret this smoke as improvement.
