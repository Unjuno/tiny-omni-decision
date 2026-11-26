# Clotho-AQA Path A CPU runtime smoke — 2026-11-26

## Scope

This adds an end-to-end inference path for the fixed **binary audio-question
probe**: local WAV + question text → frozen EfficientAT and MiniLM features →
fixed two-option readout → typed probability JSON. It does not train or alter
any checkpoint. This is an implementation/reload/parity check, not a new
quality evaluation; the six references below come from the already-exposed
internal validation partition. The publisher-validation metrics are recorded
separately in [the one-shot transfer result](R2_CLOTHO_AQA_PUBLISHER_VAL_RESULTS_20261126.md)
and were not recomputed here.

## Frozen runtime and measured parity

- MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, weights
  SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- EfficientAT source revision `a425fdce92572e602a1d5634799bd9f1f2efa806`,
  checkpoint SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`.
- Frozen readout SHA-256
  `ac3fc2012c146538537a293e007a386fd1c738ba6ee0a7452cc89ccf05daab4e`;
  normalization SHA-256
  `f152042f1f1ff03a4affb3381b5419dfe390edf64420dbd1ed178e835bbe1791`;
  temperature `1.4142135623730951`.
- Six deterministic first-distinct-source internal validation examples were
  re-encoded from WAV and question text. Maximum absolute difference from the
  saved calibrated `P(Yes)` values was `8.94e-8`. Reversing supplied `[No, Yes]`
  to `[Yes, No]` produced zero probability permutation error for all six.
- The runtime returns the repository's typed `yes_no` JSON contract and
  preserves the supplied option IDs/order. It deliberately rejects options
  other than one `No` and one `Yes`; this checkpoint is not a general
  free-form candidate scorer.

## Cost and limits

- Runtime parameters: MiniLM `17,389,824`; EfficientAT `1,961,239`; readout
  `49,409`; total `19,400,472`.
- The four pinned model/normalization files total `77,863,187` bytes, excluding
  tokenizers, source code, Python and runtime dependencies. This is not a
  packaged or quantized export measurement.
- CPU only: Python 3.11.9, PyTorch/torchaudio 2.6.0+cu124, Transformers 5.6.2,
  PyAV 18.1.0, Windows 10. A cold model load took 6.03 seconds. Six normal-order
  requests averaged 0.178 seconds each; the six reversed-order parity requests
  averaged 0.162 seconds each. This tiny parity smoke does not establish a
  latency distribution or service-level bound. No GPU was used.
- The harness and report are at
  `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-runtime-parity-20261126-01/`.
  It records per-row source/audio hashes, checkpoint pins, environment,
  parameter counts, file bytes, and exact output parity.
- This is a 19.4M-parameter audio+text task prototype. It does not include
  image, video, or sensor inputs and does not satisfy the four-modality
  product objective, 100–200M architecture study, quantized packaging gate,
  or 95–98% task-quality aspiration. The external transfer result is
  underpowered and inconclusive; do not promote this path.
