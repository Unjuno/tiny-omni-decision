# Numeric embedding coverage and technical feasibility

Training-only audit of8192 comparison-vocabulary rows /1024 unique states:
12288 numerical occurrences,583 values,343 distinct numeric subword IDs.
8259 occurrences use one subword and4029 use two. This counts the corpus, not
selected-checkpoint exposure; it does not establish that frozen embeddings cause errors.

The existing22,713,601-parameter Cross-Encoder training updates all6 encoder
blocks and classifier, leaving word embeddings frozen. A zero-initialized delta
for343 selected rows adds131712 parameters (22,845,313 total), instead of making
all11,720,448 word-embedding parameters trainable. IDs come only from training
state text; no validation/audit labels or oracle-derived features.

CPU unit test proves exact zero-init outputs, base freeze, target-row updates,
non-target-row isolation and rejection of duplicate IDs. Real adapted checkpoint
smoke loads the saved comparison-vocabulary best weights, applies one update on
the first training example, confirms exact initial logits and unchanged frozen
parameter hash. Gradient norm1.925359; no quality evaluation or adopted weights.
The one-row update is a discarded implementation check, not a learned candidate.

Local artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-embedding-coverage-01`
and `numeric-embedding-smoke-01`. Reproduce with the audit and smoke scripts using
new output directories. A first external-script launch failed before model loading
because repository imports were missing; running from repository module context
resolved it. No training artifact was overwritten.

Next evidence required: a predeclared matched training comparison with identical
initialization, unique coverage, optimizer budget and validation-only selection.
Embedding updates may help numeric representations or harm other uses of the same
tokens; measure original text and transfer rather than assuming improvement.
No full encoder/modality unfreezing, cloud compute, audit or product promotion here.
