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
model-output pooling, and mask-aware mean pooling have CPU tests over a synthetic
encoder. This verifies gradients through the adapter interface and readout, but
is not an EmbeddingGemma 2 weight forward/backward test or a quality baseline.
The temperature and input prompt template must be frozen before the first
validation comparison.

`processor_inputs_for_decision_example` resolves one local media path under the
configured data root, fails closed on remote/out-of-root/missing media, converts
audio to mono float32 16 kHz from 16-bit PCM WAV, and passes video paths with
metadata disabled for model input. `processor_inputs_for_options` preserves
option order and rejects options that become duplicates after prompt
normalization.

The pinned `AutoProcessor` was also run on CPU with CUDA hidden and synthetic
inputs; no model weights were loaded. Observed processor outputs:

| Input | Key output shapes |
|---|---|
| Text | `input_ids (1, 15)`, `attention_mask (1, 15)` |
| Image | `input_ids (1, 285)`, `pixel_values (1, 2520, 768)`, `image_position_ids (1, 2520, 2)` |
| 1 second, 16 kHz audio | `input_ids (1, 43)`, `input_features (1, 99, 128)`, `input_features_mask (1, 99)` |
| Two video frames | `input_ids (1, 280)`, `pixel_values_videos (2, 1260, 768)`, `video_position_ids (2, 1260, 2)`, `num_frames_per_video (1)` with value 2 |

These shapes confirm preprocessing paths only; they do not prove that the model
accepts the inputs or yields finite embeddings.

After adding `processor_inputs_for_decision_example`, an end-to-end temporary
media-file smoke was attempted. The process failed while importing the pinned
Processor because Windows could not load `cublas64_12.dll` (`WinError 1455`,
insufficient paging-file resources) while the existing Teacher run remained
active. It failed before creating temporary media. The helper is covered by
CPU unit tests for the text path and prompt/option-order invariants, but its
local image/audio/video file branches still need a retry when the Teacher run
and system memory pressure allow it.

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
