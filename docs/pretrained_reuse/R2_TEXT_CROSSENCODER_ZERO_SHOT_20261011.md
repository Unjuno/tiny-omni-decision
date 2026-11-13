# Frozen retrieval cross-encoder Decision diagnostic

Pinned ms-marco-MiniLM-L6-v2 was evaluated once, without training, prompt search
or temperature fitting, on the existing2,540 development decisions. It scored
9,015 question/candidate plus state pairs; explicit binary criteria were retained.
Question/candidate is the first sequence, state the second; only state is
truncated at512 total tokens (239 pairs). The overlength warning comes from
length measurement without inference; actual model inputs are bounded to512.

Accuracy .45984, NLL1.19869, Brier .64975, ECE15 .13666. CUDA execution wall time
including model load was22.32s. Loaded parameters22,713,601. This unadapted
retrieval scorer does not beat the prior criteria-preserving trained readout
(Accuracy .50906, NLL1.01490). Its logit softmax is not established calibration.
No additional training or seed selection followed this diagnostic.

This is not a controlled proof about cross-encoder architecture: the pretrained
task, depth, input representation and absence of Decision adaptation differ.
Retain as a negative zero-shot reference; do not promote it or claim general
Decision quality. No sealed data was opened.

Local directory: C:/CodexArtifacts/pretrained-reuse-r2/text-crossencoder-zero-shot-01.
The result records the executed runner hash; executed-runner.py preserves those
exact bytes. The repository runner differs only in formatting the long limitation
string after execution. Predictions and result are retained locally. Source model
revision/hashes and load audit are in R1_TEXT_CROSSENCODER_20261011.json.
