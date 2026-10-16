# Explicit event feature transport, before fitting AV policies

Use a separate managed checkout from8e90c80 while both complete caches run.
Do not change either executing checkout, corpus, encoders or saved features.
The existing FeatureStore regenerates relative sampled-order positions; retain
that historical implementation. Add a new explicit-position store for this task.

Bind complete passing audio/video receipts to frozen task rows by exact event ID,
parent, split, source media digest, source line, question count and nearest-sample
clock. Both plan and per-parent receipt ordering/index must agree. Full mode must
cover all frozen events/questions; smoke may use only the declared TRAIN subset.
Reject RUNNING/partial receipts, missing or duplicated events, cross-split or
clock mismatch. Use all original question/candidate content keys; targets do not
enter the store. Only a complete permutation may change candidate order.

Read the saved25 speech,1 event-audio and2048 video tokens with their saved position
arrays. Check finite FP32 shapes, actual video PTS/clock geometry, query identity
and file hashes. No synthetic linspace replacement, implicit pooling or missing
stream fabrication. Read-only memory maps with bounded caches; detect changed
file size/mtime and verify hashes at the start/end of each executed probe.
Joint/audio-only/video-only are explicit views of the same stored event. Frozen
encoder count195,365,529 and existing readout definitions stay unchanged.

First CPU tests using generated fixtures. Then run the previously verified fixed
21 TRAIN smoke events /63 native questions through C0 and four residual arms,
same initial tensors/seed17. Use the actual saved encoder features. No target loss,
optimizer, calibration, quality scoring or validation media. Compare baseline
parity at zero correction, genuine candidate permutations (max logit tolerance
1e-6), JSON normalization, both unimodal missing-input bypasses, exact same-policy
save/reload. Diagnostic gate values are not quality/reliability evidence.
Save each arm's state and original/permuted/joint/unimodal probabilities, hashes,
parameter ledgers and measured CPU time/memory; never reuse an output directory.
Fresh-process replay must reproduce all artifacts exactly.

Set CUDA_VISIBLE_DEVICES empty, CPU FP32 deterministic execution, one CPU thread,
seed17. No environment upgrades. Guards:15minutes,4GiB process peak working set,
16GiB disk reserve. Record concurrent audio GPU/video CPU caching. A runtime failure
stays recorded; do not weaken tensor/identity checks to pass it.

Training remains gated on complete, audited full caches and a separately frozen
learning/validation protocol. Smoke predictions from random readouts do not
measure pretrained-task accuracy or gate benefit. C0 is the original Attention;
the exchange-off/gate-off residual arm still has extra capacity and is not C0.
Preserve all original product, retention, complementary-task and OOD requirements.
