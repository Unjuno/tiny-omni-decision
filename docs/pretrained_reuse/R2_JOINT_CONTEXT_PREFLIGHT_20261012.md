# Joint context preflight

Actual local tokenizer, fixed parent training set and complete validation IDs.
CPU tokenization only. No model inference, training, label changes or filtering.
Input and runner hashes plus per-question truncation are retained locally.

| Corpus | Questions | Independent state truncated | Joint state truncated | Option overflow |
|---|---:|---:|---:|---:|
| original training | 4096 | 167 | 260 | 0 |
| curriculum training | 1024 | 0 | 0 | 0 |
| original validation | 2540 | 66 | 117 | 0 |
| curriculum validation | 2048 | 0 | 0 | 0 |

All option/question tokens fit512; state is the only truncation target. Joint
packing loses some state text in51 additional validation questions and93
additional original training questions. Even already-truncated cases may lose
more state tokens. Thus the old independent run cannot be treated as an
information-matched control for joint readout.

Before a causal readout comparison, construct both tokenizations with the same
state-token prefix length per question: minimum budget across the joint row
and every independent candidate row. Verify exact state-token identity without
decode/re-encode, all option identities/order and all evaluation IDs. This is a
controlled information-matched experiment; it is not automatically a deployable
preprocessing improvement. Record the loss of evidence. Keep original full-
context baseline metrics separately and do not describe common-prefix results
as the unchanged original benchmark. Do not remove long/difficult questions.

If improvements emerge, verify the practical complete-input path separately;
shared truncation cannot establish full-input95% quality. Chunked evidence may
be needed, but is a separate design factor and not part of this comparison.
No length extension, new training, or altered frozen corpus was performed.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r2/joint-context-preflight-01
