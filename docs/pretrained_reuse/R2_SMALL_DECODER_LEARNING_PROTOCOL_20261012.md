# Small decoder fixed learning curve

Feasibility source4adba68 passed two training updates, frozen hash and reload
checks.134,515,008total /14,160,960trainable; last4decoder blocks+final norm only.
All20,972scheduled train/validation prompts fit without truncation (max1,029,
median208,p90=434,p99=652tokens). Training peak1,619,352,576bytes; second update
0.196s. No validation quality evaluated in feasibility. Fresh base for full run;
do not carry smoke weights/optimizer/RNG forward.

One seed17,2,048updates,8examples/update (same4original+4curriculum ID order as
comparison-vocabulary run), no repeat or budget extension. AdamW2e-5/wd.01,
clip1,FP32/SDPA,model.eval/dropoutoff. Frozen first26layers/embeddings/LM head.
No new parameters, teacher distillation, media training, QAT or Recovery.

Same2,540original and2,048curriculum validation IDs/order, never truncated.
Unlike the older BERT pipeline's possible state-only truncation, full context
is retained; model/pretraining/readout/compute differ, so this is not a controlled
capacity-only comparison. Labels A..Z map to the existing complete option order.
Loss/metrics normalize over supplied labels; open-vocabulary label mass is also
reported, but is not a calibrated rejection probability.

Validate at0,256,512,...,2048. Earliest minimum mean of the two source NLLs
selects best, including step0. Preserve last checkpoint separately; no quality
early stopping or final-checkpoint substitution. Save predictions/probabilities,
per-source/task Accuracy/NLL/Brier/ECE, histories, consumed/unique/state counts,
rolling training CE (not a fixed-data gap), LR/gradient norms, time/VRAM, hashes,
trainable names and frozen hash. Require saved-best reload parity. Stop on NaN,
invalid splits/labels/hash, frozen changes, OOM or disk constraints.

New small-decoder-learning-01 output. Estimated roughly10–20minutes from smoke,
not guaranteed. Local GPU only, no cloud. All old artifacts read-only. No known
condition probes, new transfer tests or sealed audit used for selection; any
later transfer protocol must be frozen independently. Exact optimizer/RNG resume
is not supported by weights-only saves. Full Omni quality remains unproved.
