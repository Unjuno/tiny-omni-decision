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
  the manifest's pinned Git blob ID, SHA-256, and/or byte size. The full model
  weight file was downloaded outside the repository, and its SHA-256 and byte
  length match the pinned manifest.

## Checkpoint tensor inventory

The model's safetensors header was first inspected by HTTP Range requests. It
declares 1,376 BF16 tensors and
744,371,992 tensor elements (1,488,743,984 payload bytes); the remaining 171,304
bytes are file/header overhead. The complete 171,295-byte parsed header is
saved at `C:/CodexArtifacts/embeddinggemma2-tensor-inventory-914f7f8.json`
(SHA-256 `5b021dc52fea66eb3099159d7ee14c0de08960dd3c68ed011c26943b708becf7`).
This is a checkpoint storage inventory, not a trainable parameter count. After
disk availability was confirmed, the pinned weight file was downloaded to
`C:/CodexArtifacts/embeddinggemma2-metadata-914f7f8/model.safetensors`. It is
1,488,915,288 bytes with SHA-256
`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`, matching
the manifest. Loading it instantiated 744,371,488 parameter elements. The
state dict still has all 1,376 BF16 tensors; 504 scalar state tensors (480
audio metadata scalars and 24 language-model scalars) account for the
744,371,992 header element count. This distinction must be preserved when
classifying conversion targets.

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

The loaded module inventory is serialized outside the repository at
`C:/CodexArtifacts/embeddinggemma2-model-inspection-914f7f8.json` (SHA-256
`1b5ecfceea51f9dcbedf7faa5d18df3bab8ec4f9840aade4f2c991448c29ea6d`). It
records instantiated state-dict roots, weighted module paths/shapes, module
classes, scalar tensors, dtypes, and runtime versions. The graph includes 467
ordinary `Linear` modules and two `Conv2d` modules with direct weights. The
audio path also has 12 custom `Gemma4AudioCausalConv1d` modules with direct
weights. Weighted modules also include 204 `Gemma4RMSNorm`, 170
`EmbeddingGemma2RMSNorm`, one scaled word embedding, and two LayerNorm modules.
A linear-only converter would omit audio convolution tensors and embedding
weights, so the serialized tensor inventory must drive conversion and every
exception must be accounted for.

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
encoder. A separate real-checkpoint CPU forward/backward smoke is recorded
below; neither the synthetic tests nor that one synthetic text example is a
quality baseline. The temperature and input prompt template must be frozen
before the first validation comparison.

`processor_inputs_for_decision_example` resolves one local media path under the
configured data root, fails closed on remote/out-of-root/missing media, converts
audio to mono float32 16 kHz from 16-bit PCM WAV, and passes video paths with
metadata disabled for model input. `processor_inputs_for_options` preserves
option order and rejects options that become duplicates after prompt
normalization.

The pinned `AutoProcessor` was run on CPU with CUDA hidden and synthetic
inputs. Initially no model weights were loaded; a later text smoke loaded the
full pinned checkpoint. Observed processor outputs:

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
active. It failed before creating temporary media. After disk and system memory
became available, the image, audio, and one-frame video local-file paths were
exercised against the real checkpoint on CPU. The first audio attempt exposed a
closed-WAV-handle bug; a regression test now covers it and the implementation
reads PCM frames before closing the file.

## Actual checkpoint CPU smoke

On 2026-10-07, the full pinned checkpoint loaded with Python 3.11.9, PyTorch
2.6.0+cu124, Transformers 5.19.0, BF16 weights, eager attention, and CUDA
hidden from the process. The existing Teacher process remained running; no
student GPU allocation or optimizer update was made. Processor loading took
about 2.04 seconds and model loading about 3.68 seconds after Python startup.
The CPU was a 12th Gen Intel Core i7-12700H (14 cores / 20 logical processors);
the smoke processes used four Torch CPU threads. The RTX 3080 Laptop GPU had
16,167 MiB of 16,384 MiB allocated to the Teacher, so the student stayed on
CPU. The video processor used torchvision 0.21.0+cu124 because torchcodec was
not installed; PyAV 18.1.0 was available but not selected by this processor
path.

A synthetic text batch of one query and two options produced finite normalized
embeddings of shape `(3, 768)`. With temperature `0.1`, option logits were
approximately `[9.4648, 9.4500]`; CE plus one-hot Brier loss was `1.17841`.
Backward completed in about 24.02 seconds. The language-model embedding
projection gradient was finite with norm `0.017726`. The process performed no
optimizer step and modified no weights.

The actual checkpoint also completed no-grad inference on locally generated
media files:

| Modality | Processor input | Output | CPU time | RSS at completion |
|---|---|---|---:|---:|
| Image | `pixel_values (1, 2520, 768)` | finite `(1, 768)` embedding | 162.10 s | 2.38 GB |
| Audio | `input_features (1, 99, 128)` | finite `(1, 768)` embedding | 2.26 s | 2.36 GB |
| Video, one frame | `pixel_values_videos (1, 1260, 768)` | finite `(1, 768)` embedding | 49.11 s | 2.39 GB |

These are plumbing tests with synthetic files, not benchmark examples. The
video processor warned that torchcodec was unavailable and used its torchvision
fallback decoder. `model_sentence_embeddings` moves processor tensors to the
model input-embedding device while preserving dtype; a unit test exercises
that transfer helper without requiring CUDA.

These are real-checkpoint text and synthetic-media plumbing smokes, not
accuracy or calibration results. They do not evaluate corpus media, calibrate
the temperature, or measure ternary behavior. No quality threshold or
checkpoint selection is based on these synthetic samples.
The full machine-readable report is at
`C:/CodexArtifacts/embeddinggemma2-multimodal-smoke-914f7f8.json` (SHA-256
`2195f8bc99f3432e97e245fb7ad1475841cef8d2227fe0c6593ae073029e0b52`); the
actual model graph inventory is in the separately hashed report above.

## Phase 6 status

Completed: exact model/processor pin and license metadata; full checkpoint
download with hash verification; config, processor, and model loading with
Transformers 5.19.0; actual tensor/module inventory; Sentence Transformers
pooling metadata; and one finite CPU forward/readout/backward smoke through
the real checkpoint.

Also completed: synthetic local-file image, audio, and one-frame video
inference through the actual checkpoint on CPU, with finite 768-dimensional
outputs. Not yet completed: measuring unquantized same-input quality on the
frozen validation set, freezing quality and deployment limits, ternary
conversion, packed export, or runtime verification. The active Gemma 4 Teacher
run still occupies the shared GPU, so all student smokes used CPU and no
student GPU workload was started. No ternary size or quality result is
claimed.
