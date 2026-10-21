# Compact joint-candidate plumbing

Independent experimental implementation; no D1 source copied or weights used.
Existing pinned MiniLM CrossEncoder + coverage1792 delta;22,713,601 parameters,
zero added. Tokenize question and all options jointly, preserve every option,
truncate only the state if needed. One contextual separator marker per option;
reuse existing BERT pooler dense/tanh and scalar classifier at each marker.
This differs from trained CLS readout and requires training before quality use.

Real GPU smoke: two questions,2/3options,71/78tokens; no truncation. Repeated
logits exact. Warm two-question batch7.154ms, peakallocated102,366,208bytes.
One warm measurement, not p50/p95 or a speed comparison with independent scoring.
JSON round trip preserves candidate identities and probabilities. Source delta
hash unchanged. No training, benchmark/audit evaluation, or product adoption.
One of two plumbing questions is wrong: do not interpret successful execution
as useful zero-shot reasoning or estimate accuracy from these two fixtures.

Tests cover full option retention, state-only truncation, context overflow,
variable widths, finite outputs, gradient reachability and unchanged parameter
count. Candidate-order invariance is NOT guaranteed with positional embeddings;
a controlled learning comparison must include order perturbations. Sharing one
question sequence does not cache the trunk across distinct questions.

Next: freeze task/sample order and training budgets for joint versus independent
readout; retain all modalities' goals, use validation-only selection, separate
outputs and no sealed audit. Joint sequence changes both contextual interaction
and compute; record both, and reject context/option loss rather than silently
simplifying the task. This smoke does not justify adopting an architecture.

Artifact: C:/CodexArtifacts/pretrained-reuse-r2/joint-candidate-smoke-01
