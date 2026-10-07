# EmbeddingGemma 2 Student

This records the first Phase 6 implementation evidence on branch
`codex/ternary-student-recovery`. The source roadmap is `origin/main` at
`09246d7b1821662a6b1eba47fc08c43f9dc3863c`; the model and processor are pinned
in [the student manifest](../manifests/embeddinggemma2-student.yaml).

## Revision and implementation compatibility

- Model and processor: `google/embeddinggemma-2`, revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- License metadata: Apache-2.0; the upstream model card is linked from the
  manifest.
- The repository's shared environment has Transformers 5.6.2, which does not
  register `embedding_gemma2`. Transformers 5.19.0 successfully loads the pinned
  `EmbeddingGemma2Config` and, with the pinned tokenizer files, the
  `EmbeddingGemma2Processor` and `GemmaTokenizer` (vocabulary size 262,144).
- The new `[student]` extra isolates that compatibility range from the existing
  Teacher-oriented `[ml]` extra. Do not combine the extras in one installation;
  the active Teacher environment remains unchanged.
- Every downloaded metadata, tokenizer, and pooling/normalization file matches
  the manifest's pinned Git blob ID, SHA-256, and/or byte size. The model weight
  file itself remains undownloaded; its SHA-256 and length are pinned from Hub
  metadata, while its tensor header was read with HTTP Range requests.

## Checkpoint tensor inventory

The model's safetensors header was inspected by HTTP Range requests. The 1.49 GB
weight file was not downloaded or opened. It declares 1,376 BF16 tensors and
744,371,992 tensor elements (1,488,743,984 payload bytes); the remaining 171,304
bytes are file/header overhead. The complete 171,295-byte parsed header is
saved at `C:/CodexArtifacts/embeddinggemma2-tensor-inventory-914f7f8.json`
(SHA-256 `5b021dc52fea66eb3099159d7ee14c0de08960dd3c68ed011c26943b708becf7`).
This is a storage inventory, not a trainable parameter count.

| Tensor root | Tensors | Elements | BF16 payload bytes |
|---|---:|---:|---:|
| `language_model` | 413 | 271,002,648 | 542,005,296 |
| `vision_tower` | 210 | 167,364,608 | 334,729,216 |
| `audio_tower` | 751 | 304,825,088 | 609,650,176 |
| `embed_vision` | 1 | 393,216 | 786,432 |
| `embed_audio` | 1 | 786,432 | 1,572,864 |

The audio root contains 480 scalar tensors used as min/max metadata as well as
the main weight tensors. A conversion implementation must classify every tensor
by its actual name and role; scalar metadata is not silently treated as a
matrix weight. No quantization target or exception has been selected yet.

## Minimal supplied-option readout

The pinned Sentence Transformers metadata defines mean pooling over the model's
projected token outputs (including prompt tokens), followed by L2 normalization.
The current candidate readout uses that native 768-dimensional embedding path.
For the symmetric query/option comparison, both sides use the model card's
`task: sentence similarity | query:` instruction prefix:

1. Encode the state/question and any attached image, audio, or video as the
   query input.
2. Encode each supplied answer option as text with the same prefix in the same
   processor/model.
3. Score each option by query/option cosine similarity divided by an explicit
   temperature; preserve the source option order and map the target to its
   original option index.

The placeholder for attached media is placed after the query text and paired
with the corresponding processor input. No option text is included in the query
representation. This adds no classifier or output vocabulary, and does not
interpret Teacher token IDs as student token IDs. The readout, prompt builder,
and mask-aware mean pooling have CPU tests over synthetic vectors. The backward
test validates score math and gradients only; it is not a model
forward/backward test or a quality baseline. The temperature and input prompt
template must be frozen before the first validation comparison.

## Phase 6 status

Completed: exact model/processor pin, license metadata, config and processor
loading with Transformers 5.19.0, Sentence Transformers pooling metadata, and a
weight-file tensor inventory from its header.

Not yet completed: loading the checkpoint weights, inspecting instantiated
weight paths/module shapes, running multimodal forward/backward, measuring
unquantized same-input quality, freezing quality/deployment limits, converting
to ternary, or verifying an export/runtime. The active Gemma 4 Teacher run is
still using the shared GPU; no GPU workload was started for this student work.
The checkpoint weights exceed currently free local disk, so they were not
downloaded. No ternary size or quality result is claimed.
