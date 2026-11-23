# LINE DistilBERT long-pair smoke — 2026-11-24

## Result

The BERT candidate can encode the longest question/option pair found in the
fixed native common corpus without dropping any of its 516 tokens. The frozen
four-layer encoder emitted a finite 768-wide vector on CPU. This is a feature
transport check only; it is not a Decision accuracy result and does not qualify
the BERT tower for product use.

## Why this check was needed

The common corpus contains 24,950 unique question/option pairs. The pinned LINE
Japanese DistilBERT tokenizer gives the longest pair 516 tokens, above the
encoder's 512-token context. Rejecting the pair would leave a required source
record without features; truncating it would alter the option text. The updated
feature function instead uses 512-token windows with a 32-token overlap and
retains every token position exactly once for mean pooling. Inputs above 8,192
tokens fail closed.

The smoke used pair-key SHA-256
`a38f9c5336638a4ab55b50d6ceb0c97899cdbc8915ecdcfdc109e02cd75a4d83` and
produced window lengths `[512, 36]`, retaining all 516 positions.

## Model and runtime

- Model: `line-corporation/line-distilbert-base-japanese`, revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`, Apache-2.0.
- Source checkpoint `pytorch_model.bin` SHA-256:
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
- The source checkpoint has six layers; the candidate explicitly retains the
  first four and has 53,912,064 total parameters, all frozen.
- The source is a masked-language-model checkpoint loaded as `DistilBertModel`.
  The six MLM-head tensors are reported as unexpected and intentionally
  discarded; there are no missing or mismatched encoder tensors.
- CPU, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  tokenizers 0.22.2; the operation took 3.78 seconds.
- Output shape `[768]`, finite FP32 values. Encoder state digest was unchanged.
- Receipt:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\line-distilbert-longpair-smoke-20261124-attempt-01\receipt.json`.
  Output feature SHA-256:
  `d600a66cd8188d0c6eb4438b6be6b583e209395a7e3a0209195299a595e46bcb`.

## Code verification and limits

- Focused tests: 28 passed; Ruff passed.
- The working source was dirty during the standalone smoke. The receipt records
  the source commit and flags this; it must not be described as a clean-commit
  reproduction.
- The smoke did not train a readout, load validation labels, evaluate quality,
  or load the sealed audit.
- A label-free cache-plan pass over the pinned source inventory found 4,382
  unique text states and all 24,950 common-corpus option queries. The split
  feature shapes are `[497588,768]` for train states,
  `[321770,768]` for validation states, and `[24950,768]` for query vectors.
  FP32 storage is 2,593,714,176 bytes (about 2.42 GiB). The longest state is
  1,318 tokens; 248 states exceed 512 tokens. One query pair is 516 tokens.
  All remain below the 8,192-token guard. State-order SHA-256 is
  `78330026a1ff78d17829475f428da64f38b6982931a5d57ff2a4859d4fee51e3`;
  query-order SHA-256 is
  `e5009121b18d4d92c06c22f4e4d6b3774953c25a4713d039257b5ed47f483622`.
- The storage/tokenization counts were first checked before capture. The later
  full capture outcome is recorded below; neither phase produces quality
  metrics.

## Full frozen feature-cache capture

The source/config/cache runner was committed and pushed as `b6c64baf` before
execution. Its smoke receipt matched the exact commit, model, corpus, tokenizer
runtime, and environment. The full cache then completed:

- 4,382 / 4,382 text states and 24,950 / 24,950 query pairs encoded.
- Train state bank `[497588,768]`, validation state bank `[321770,768]`, and
  query bank `[24950,768]`; total 2,593,714,176 FP32 bytes.
- GPU: RTX 3080 Laptop, peak allocated 291,699,712 bytes and peak reserved
  312,475,648 bytes. State encoding 26.77 seconds, query encoding 109.49
  seconds, total run 155.32 seconds.
- Model parameter digest before and after was identical:
  `26a50f8980f2724ebd7e40b24f6996e5ab1332bd26a0eda888b17bac196c18c0`.
- Cache result receipt SHA-256:
  `0d30d3a05b43918456233044229075a234f685527b31590c9609818bbc9d6f84`.
- Output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\path-a-line-distilbert-bert-features-cache-20261124-attempt-01\`.

A separate CPU `FeatureStore` overlay smoke validated the new text states and
all 24,950 query vectors while preserving the existing image, video, speech,
and event-audio feature banks. One training row per modality returned finite
query features with the expected option counts. This proves feature-overlay
transport only; no readout training or quality evaluation has happened yet.
- A 768-wide BERT cache and same-protocol Path A comparison remain necessary.
  Existing common-corpus validation is already exposed, so a result there is
  development evidence, not blind confirmation.
- The broader BERT-family reconciliation remains the governing evidence:
  BERT-family encoders help on some bounded text tasks, but do not establish
  broad option reasoning or multimodal fusion quality. Avoid another open-ended
  model sweep; continue with the one pinned BERT Path A integration candidate.
