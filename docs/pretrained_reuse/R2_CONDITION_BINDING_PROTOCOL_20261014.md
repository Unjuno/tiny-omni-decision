# Fixed train-only question/state binding diagnosis

Follow the completed spatial appearance comparison; the complete Omni goal
remains unchanged. Systematic debugging is used to test the suspected loss of
question condition before proposing another training change. User authorization
permits autonomous bounded inference-only diagnostics; no repeated approval.

## Fixed probe

Use only existing temporal_appearance best(epoch6) and final(epoch50), no new
weights/learning/checkpoint selection. Preserve all99 original train videos,
198questions/3options and53 changed-target pairs. The common integrity loader
validates old train/development caches but **model evaluation is train only**.
No development predictions, sealed/audit data, new corpus or encoder changes.

Four predeclared conditions, with original targets unchanged:

1. Normal: exact original saved train output replay.
2. Neutral question: replace both question384-d vectors by their input-only
   average, preserving each option vector; rebuild question/option/absolute
   difference/product1536-d features. Identical question inputs cannot satisfy
   two different targets; this control is not an alternative deployable task.
3. Swapped question: swap question vectors, keep option vectors/state unchanged;
   rebuild features. Check logits/probabilities against the normal paired outputs
   in reversed condition order, proving candidate/condition routing consistency.
4. Mismatched whole state: cyclically replace both point motion/initial appearance
   and spatial context with another training video's state, preserving question/
   options/target. This differs from the prior context-only mismatch and measures
   state dependence, not correct visual reasoning or valid new-source quality.

First verify exact cached text factorization, two distinct question vectors,
option equality between conditions and ordered sample/target provenance. Measure
question-vector cosine/norm/difference; do not infer quality from cosine alone.
Return original QA Accuracy/NLL/Brier/ECE15, condition-pair and changed-target
both-correct, answer/condition changes and paired logit/probability perturbations.
Record hashes/revisions/source commit, inference time, memory and all attempts.
Require original epoch archive SHA before shape/dtype/tobytes replay. Save raw
logits and measure both raw and per-row-centered logit changes. Question-swap
logit/probability gates use the existing candidate-equivariance tolerance
rtol1e-5/atol1e-6; normal original-epoch replay remains strict byte equality.
Every registered readout parameter is frozen, before/after state hashes equal,
CPU/CUDA RNG unchanged, all outputs finite; cap total600seconds/4GiB allocation,
12GiB free GPU preflight. Preserve original best/final/config/run files.

## Interpretation

If final training predictions depend on the question and state while selected
best collapses to an option prior, condition information/routing loss is not a
sufficient explanation of failure. That supports investigating generalizable
object/cover association and task representation rather than more repetition.
It does not prove a unique cause or general capacity ceiling. No control metrics
choose new weights or revise held-out quality; no product promotion/quantization.
