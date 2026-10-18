# CLEVRER temporal task-mix BiLSTM protocol (2026-10-18)

## Question

Does preserving the eight ordered VideoPrism frame-position features in a small
bidirectional LSTM help a frozen-feature Decision readout beyond the same
question-and-candidate text alone? This is a scoped video experiment, not a
claim about the complete Omni model or product quality.

## Frozen data

- Source: MIT-IBM/CLEVRER training questions at revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0).
- Seed: 17. The deterministic selector excludes previously observed scene
  identities, then groups the remaining records by underlying video scene.
- Training split: 1,024 rows from 614 scenes, with realized mix
  temporal descriptive 204, explanatory 306, predictive 306, and
  counterfactual 208 rows. (The source mix is intended to be 20/30/30/20; the
  recorded realized count is authoritative.)
- Validation split: 1,400 rows from 800 disjoint scenes; 200 scenes per task
  type. Reasoning parents contribute one correct and one wrong candidate row.
- The selector enforces scene, parent-question and normalized question/options
  separation. Media materialization verifies per-video SHA-256 and split
  identity. No sealed-audit data are loaded.
- Split and media manifest hashes are saved in the external preflight and run
  metadata; model weights and media remain outside Git.

## Input representation

For every reasoning candidate, the MiniLM input is the original CLEVRER question
plus the candidate statement. It is not inferred from the binary label. The
temporal descriptive examples retain their original multi-choice question and
taxonomy options. MiniLM-L3 and VideoPrism are frozen. VideoPrism supplies eight
ordered frame-position features; the candidate model is a 128-hidden-size
bidirectional LSTM followed by a shared candidate MLP. A text-only candidate MLP
is the matched reference. Both models score the same frozen train and validation
rows and return softmax probabilities over each row's supplied options.

The planned VideoPrism+MiniLM+BiLSTM+candidate-readout total is 133,100,801
pre-quantization parameters (VideoPrism 114,365,184; MiniLM 17,389,824; trainable
readout 1,345,793), under the provisional 200M target. This accounting is for
this experiment's component only, not the full product model.

## Fixed fit and selection

- Seed 17; fresh readout initialization for each arm.
- AdamW, learning rate 0.001, weight decay 0.001.
- 64 epochs, batch size 32 questions, no performance early stopping; at most
  2,048 optimizer updates per arm.
- Each frozen training row is visited once per epoch, shuffled with seed 17.
- Checkpoint selection is minimum validation macro question-type NLL; strict
  ties retain the earlier epoch. Final and selected checkpoints are saved
  separately, and the selected checkpoint is reloaded and compared exactly.
- Every epoch records train/validation Accuracy, NLL, Brier and ECE by task.
- A paired validation scene bootstrap compares the selected VideoPrism+BiLSTM
  arm with text-only, stratified by task type (10,000 replicates, seed 17).
  Because validation selects the checkpoints, these intervals are descriptive
  and are not independent-test guarantees.

## Compute and safety

Use only the local RTX 3080 Laptop; no cloud compute or purchases. GPU work is
launched behind a cooperative GPU-yield gate. It waits for a stable idle GPU,
then reserves 3.2 GiB so the reported 14,000 MiB free-memory gate for the
Orbitune supervisor remains closed while the child job uses the remaining
memory. It allows at least 8 GiB for the child and refuses to launch if those
limits cannot both be met. It does not stop or modify Orbitune. If the GPU does
not yield, the job does not launch. Sealed audit, existing Teacher artifacts
and prior runs remain read-only.

## Results

Pending local media verification, frozen VideoPrism feature extraction, and the
two matched readout fits. Do not interpret the protocol or preflight as a model
quality result.
