# Path A LINE DistilBERT text-lens integration smoke — 2026-11-24

## Result

The Path A native readout now has an optional shared 768-to-384 text lens.
With `text_input_width=768`, the same learned lens transforms both text-memory
tokens and question/option vectors before the existing native readout. Image,
video, speech, and event-audio interfaces are unchanged. The default
384-dimensional path uses `nn.Identity`, so the original readout parameter
shapes and state-dict keys remain compatible.

The actual pinned four-layer LINE Japanese DistilBERT checkpoint was loaded
locally on CPU, frozen, and passed through the repository's token-feature
functions and the Path A candidate scorer. The smoke produced finite 768-wide
observation and candidate features, finite option logits, and exactly identical
logits after readout save/reload. It evaluated one hand-written input only; the
untrained scorer's output is **not a quality metric**.

## Fixed implementation and budget

- Model: `line-corporation/line-distilbert-base-japanese`, revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`, Apache-2.0.
- Starting artifact: pinned six-layer local checkpoint SHA-256
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`;
  layers 4–5 are omitted before inference. The resulting four-layer encoder
  has 53,912,064 parameters and a 768-wide hidden state.
- New shared text lens: 768 × 384 + 384 = 295,296 trainable parameters.
- Path A readout plus lens: 1,184,641 trainable parameters in the smoke.
- Replacing the 70,830,722-parameter DeBERTa entry in the prior
  196,254,874-parameter native composition gives an **arithmetic projection**
  of 179,631,512 total pre-quantization parameters. This assumes all four
  native modality paths use the replacement text tower and does not count a
  packaged runtime or establish deployment size.

## Execution evidence

- Existing R0–R8 roadmap branch base: `32767c281de0de1504973e8a79de37621f9c2fc6`.
- Smoke source branch started from the one-shot KUCI record commit
  `187cb33c3681209cadacf5ba47dcc4ac30c0b87d`; the smoke used the local dirty
  integration source, whose exact file hashes are recorded in the companion
  manifest after commit.
- CPU-only, local checkpoint/tokenizer, no network, no GPU, no training, no
  dataset rows, no validation, and no sealed audit were loaded.
- Observed shapes: one state `[10, 768]`, four candidate vectors `[4, 768]`.
- Readout probabilities were finite and normalized; reloaded logits matched
  exactly. These are runtime-integrity checks, not model-quality evidence.
- Focused CPU tests: 45 passed. Ruff: passed.
- No pretrained weights or data were copied into the repository.

## Next gate and limits

The feature-bank capture and training runner still need a separately pinned
768-wide text corpus/cache configuration. The existing 384-wide cache cannot be
reused as BERT features. Build that cache in a new external output directory,
leaving all previous banks read-only; then run one fixed Path A comparison.
The existing common-corpus validation has already been observed, so any such
score is exploratory development evidence rather than blind confirmation. Do
not use the opened KUCI/Jamp publisher splits to tune the integrated model.

This smoke does not demonstrate that BERT improves fused multimodal Decision
quality, does not change product selection, and does not meet the 95–98%
quality aspiration. Image, audio, video, joint-input, OOD, calibration,
packaged size, and target-device gates remain open.
