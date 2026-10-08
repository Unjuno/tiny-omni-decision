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

The first fail-closed selector candidate was then run against the instantiated
checkpoint. It selected 483 tensors / 744,129,664 elements (1,488,259,328 raw
BF16 bytes): 467 Linear, 2 Conv2d, 12 custom audio causal Conv1d weights, the
scaled text embedding weight, and the exact vision patch positional table.
It matched the separately inspected inventory with no unclassified matrix
parameter. The remaining high-precision parameters are 241,824 elements
(483,648 BF16 bytes): 376 normalization weights, 12 audio `per_dim_scale`
vectors, and one audio output-projection bias. The 504 scalar state tensors
(1,008 BF16 bytes) are buffers, not parameters; runtime buffers also need
native regeneration or explicit serialization handling. The machine-readable
inventory is outside Git at
`C:/CodexArtifacts/embeddinggemma2-quantization-target-inventory-914f7f8.json`
(SHA-256 `a0b4fad68f4946a941e3a58adce3da35f2d9a814117675391e177179b33f29b5`).
This is an architecture coverage candidate, not a quality-selected target
policy, conversion, packed artifact, or size result.

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
before the first validation comparison. `evaluate_student_examples` in
`src/tiny_omni_decision/student_eval.py` evaluates the examples in supplied
order, records each original option list and its SHA-256, and returns
Accuracy/NLL/Brier/ECE by modality and source plus macro-modality and minimum
modality Accuracy. Its CPU tests use a synthetic encoder only. No frozen
corpus validation metrics have been produced yet; keep the test/evaluation
corpus out of model selection.

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

## Fixed validation comparison set

The frozen durable corpus has 2,901 validation examples. Teacher v1's actual
run metadata reports that checkpoint selection measured 256 of them: 64 per
modality. Its ignored local `best-validation-predictions.jsonl` contains those
256 records. Matching those IDs against the read-only frozen validation JSONL
confirmed exact prediction order, matching targets/source/modality, and
matching option counts. A derived snapshot outside Git is stored at
`C:/CodexArtifacts/embeddinggemma2-validation-v0-256/validation.jsonl`
(SHA-256 `43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`).
Its checked-in selection manifest is
[embeddinggemma2-validation-v0-256.json](../manifests/embeddinggemma2-validation-v0-256.json)
(SHA-256 `7aa1168ce3baa6402e0ed05b21038015c8dcc3607570e942f3a92ad667a9bc08`).
The ordered selected-ID hash is
`1334b45f6910b7b104241324163fca6a6357f1fff503718a59201fda251b8bd4`.

This is a validation-only paired comparison subset chosen by the existing
Teacher checkpoint-selection procedure, not by Student outcomes. The 2,917
record final evaluation and any sealed audit were not read. The unquantized
Student baseline will first be compared with Teacher on these exact 256 IDs and
option orders. The remaining 2,645 validation examples are not silently
substituted into that paired Teacher comparison.

`tiny-omni-student-evaluate` verifies the model file hashes, validation
snapshot/source/Teacher hashes, IDs, targets, option counts, and option-order
hashes before inference. It writes each prediction incrementally and resumes
only from an exact validated prefix. The baseline temperature is explicitly
passed on the command line; use `0.1`, the setting used in the earlier
synthetic readout smoke, without fitting temperature on this validation run.
The CLI defaults to CPU. CUDA requires explicit selection and at least 7 GiB
free VRAM, preventing overlap with the running Teacher. Pass the repository's
top-level `data` directory as `--data-root`; the frozen JSONL media references
are relative to that directory, not the processed-corpus directory.

## Phase 6 status

Completed: exact model/processor pin and license metadata; full checkpoint
download with hash verification; config, processor, and model loading with
Transformers 5.19.0; actual tensor/module inventory; Sentence Transformers
pooling metadata; and one finite CPU forward/readout/backward smoke through
the real checkpoint.

Also completed: synthetic local-file image, audio, and one-frame video
inference through the actual checkpoint on CPU, with finite 768-dimensional
outputs. The verified 256-example same-input unquantized evaluation is in
progress; its partial predictions remain outside Git and no complete quality
metrics are available. The active Gemma 4 Teacher run still occupies the
shared GPU, so student work remains CPU-only.

## Packed overlay and runtime round-trip smoke

On 2026-10-08 JST, the pinned local BF16 checkpoint was loaded with Transformers
5.19.0 and PyTorch 2.6.0+cu124 on CPU, one thread, eager attention. The complete
483-tensor candidate inventory (744,129,664 elements) was run through the
ternary-QAT parametrization. Its weights were restored to their original base
values, exported as five-trit packed codes plus FP32 scales, and applied to a
fresh load of the same hash-verified base. The fresh load's synthetic query and
two-option embeddings exactly matched the QAT forward outputs (`torch.equal`;
embedding SHA-256
`81a32bc804888f05ba6658a384e3bc9726cf198766ba896b1d8abe8b58936a0d`). The
export took 23.91 seconds; fresh reload and forward took 20.16 seconds; the
whole smoke took 99.25 seconds.

The run artifact is outside Git at
`C:/CodexArtifacts/embeddinggemma2-ternary-overlay-v0/`. Its packed
`weights.safetensors` is 160,577,615 bytes: 148,826,163 code bytes plus
11,627,028 scale bytes, with the remainder in the safetensors header. The
overlay contains 483 targets / 744,129,664 elements and retains a dependency on
the pinned base. The manifest lists every target shape/dtype and all exceptions:
483,648 bytes of high-precision parameters and 5,258 bytes of runtime buffers.
The checkpoint file remains external and unchanged. The manifest SHA-256 is
`01e4d9e6b265c149f3aeefc25775f6a1e5d4ed1fb224f345339872edfe5e0997`; the tensor
file SHA-256 is
`481be94149c6784bcc052e21503ae160f11c8971972024a09ebba0e911b4a4c1`.
The machine-readable runtime report is `runtime-roundtrip-report.json`,
SHA-256 `63250285ece72e517916a0143ed8f37e2ba964760abaf8e32dcc05f2458b1aee`;
it records working-tree source hashes and exact run metadata. The external
smoke driver SHA-256 is
`a1d1dd452064528bbeecc9015aebd4c81ecd2901687bf345478eb9bf14c38574c`.

The successful command used the existing isolated Transformers site-packages
with the shared PyTorch runtime; no dependency was upgraded:

```powershell
$env:PYTHONPATH='C:\CodexArtifacts\venvs\embeddinggemma2-v518\Lib\site-packages;src'
python 'C:\CodexArtifacts\run_embeddinggemma2_ternary_overlay.py' --model-path 'C:\CodexArtifacts\embeddinggemma2-metadata-914f7f8' --model-manifest 'manifests/embeddinggemma2-student.yaml' --output-dir 'C:\CodexArtifacts\embeddinggemma2-ternary-overlay-v0'
```

This verifies architecture-aware conversion, packed serialization, integrity
checks, and the standard Transformers load path after BF16 dequantization. It
does not verify a runtime that computes directly from packed codes, and it
does not reduce inference RAM. The five-trit code payload is physically 1.6
bits per target weight before scales, headers, exceptions, and model metadata;
this is not a whole-model storage claim. A BF16 edge case also surfaced during
the smoke work: the previous identity-STE expression could round away from the
ternary value in the forward pass. It now adds a zero-valued gradient term to
the detached quantized tensor, and a regression test checks exact BF16 forward
codes plus identity gradients.

The standard evaluator accepts `--ternary-overlay` and includes the overlay
manifest hash in run identity. The initial CPU baseline attempt was slow and
was stopped after 112 durable rows when the same fixed validation was completed
on the idle RTX 3080 Laptop GPU. Its partial output remains external and is not
used as a result. The completed GPU runs and paired metrics follow. No sealed
audit data was read. The CPU test suite, Ruff, and manifest validation had
passed before these evaluation-only documentation changes.

## Phase 7 — Initial quantization damage measurement

On 2026-10-08, the unquantized pretrained checkpoint and packed ternary
overlay were evaluated sequentially on the same fixed validation snapshot,
with identical sample IDs, targets, option strings/order, processor, BF16
weights, eager attention, temperature `0.1`, and CUDA device. The snapshot has
256 examples (64 each for text, image, audio, and video), SHA-256
`43358f0e4444dabfaba30eba0d9e4af7851f1aff56d71b2edbf38982b1bb6faf`. The
selection manifest SHA-256 is
`7aa1168ce3baa6402e0ed05b21038015c8dcc3607570e942f3a92ad667a9bc08`. Every
prediction record's sample ID, target, options, and option-order hash matched
between runs. The Teacher column is its already-saved prediction on those same
validation examples; it is a reference, not a new run or a sealed score.

| Modality | Teacher accuracy / NLL / Brier / ECE | Unquantized student | Unrecovered ternary | Ternary minus unquantized accuracy |
|---|---:|---:|---:|---:|
| Audio (n=64) | 0.9219 / 0.2283 / 0.0766 / 0.0454 | 0.5938 / 2.1284 / 0.8607 / 0.4684 | 0.0938 / 2.6073 / 0.9629 / 0.1451 | -0.5000 |
| Image (n=64) | 0.5938 / 1.2357 / 0.5422 / 0.1141 | 0.4688 / 2.1178 / 0.8565 / 0.3335 | 0.1094 / 2.3543 / 0.9119 / 0.0832 | -0.3594 |
| Text (n=64) | 0.7188 / 0.5733 / 0.3470 / 0.0933 | 0.4062 / 1.0546 / 0.6345 / 0.1942 | 0.2969 / 1.1238 / 0.6777 / 0.2907 | -0.1094 |
| Video (n=64) | 0.4062 / 1.2781 / 0.6607 / 0.1242 | 0.3125 / 1.3226 / 0.6823 / 0.0812 | 0.2656 / 1.4057 / 0.7247 / 0.1253 | -0.0469 |
| Overall (n=256) | 0.6602 / 0.8288 / 0.4066 / 0.0490 | 0.4453 / 1.6559 / 0.7585 / 0.2412 | 0.1914 / 1.8728 / 0.8193 / 0.1495 | -0.2539 |

Metric tuple order is accuracy / NLL / Brier / ECE. Macro-modality metrics were:
unquantized `0.4453 / 1.6559 / 0.7585 / 0.2693`, ternary
`0.1914 / 1.8728 / 0.8193 / 0.1611`. The lower ternary ECE does not mean
better task quality: accuracy, NLL, and Brier all worsened. Per-source metrics
are in the external `metrics.json` files and summarized below.

| Source | n | Unquantized accuracy / NLL / Brier / ECE | Ternary accuracy / NLL / Brier / ECE |
|---|---:|---:|---:|
| MIT-IBM/CLEVRER | 64 | 0.3125 / 1.3226 / 0.6823 / 0.0812 | 0.2656 / 1.4057 / 0.7247 / 0.1253 |
| TypeSafeAI/Open-Jev | 32 | 0.2500 / 0.9587 / 0.6262 / 0.3052 | 0.1875 / 1.0104 / 0.6884 / 0.4456 |
| google/speech_commands | 64 | 0.5938 / 2.1284 / 0.8607 / 0.4684 | 0.0938 / 2.6073 / 0.9629 / 0.1451 |
| n4ze3m/typed-decisions-synth | 32 | 0.5625 / 1.1504 / 0.6428 / 0.1941 | 0.4062 / 1.2373 / 0.6669 / 0.1580 |
| sgvaze/clevr4 | 64 | 0.4688 / 2.1178 / 0.8565 / 0.3335 | 0.1094 / 2.3543 / 0.9119 / 0.0832 |

Across examples, top-1 prediction agreement was `0.28125`; mean per-example
Spearman correlation of option-probability ranks (average ranks for ties) was
`0.04821`; mean KL(unquantized || ternary) was `0.14088` nats and mean
Jensen-Shannon divergence was `0.03250` nats. These indicate substantial
representational change, not stable rank-preserving compression. The
unquantized comparison is itself an untrained pretrained-embedding/cosine
readout diagnostic, so this conversion-damage result does not estimate the
loss of a subsequently trained student.

Both runs used `google/embeddinggemma-2` revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, Transformers `5.19.0`, PyTorch
`2.6.0+cu124`, CUDA `12.4`, BF16 weights, eager attention, and an NVIDIA RTX
3080 Laptop GPU (16 GiB). Baseline duration was `121.32 s`; ternary duration
was `130.47 s`. Free VRAM at each run start was `14.91 GiB`. A live `nvidia-smi`
sample while the models were loaded showed about 2.8–3.0 GiB in use; actual
peak VRAM was not captured and is UNKNOWN. No cloud compute or compute cost was
used. The stock Transformers runtime dequantizes the overlay to BF16; these
timings establish neither packed-kernel speedup nor inference-memory reduction.

Run directories remain outside Git:

- Baseline: `C:/CodexArtifacts/embeddinggemma2-unquantized-validation-v0-256-cuda/`
- Ternary: `C:/CodexArtifacts/embeddinggemma2-ternary-validation-v0-256-cuda/`
- Incomplete CPU prefix (112 rows, not a result): `C:/CodexArtifacts/embeddinggemma2-unquantized-validation-v0-256/`

| Run | Predictions SHA-256 | Metrics SHA-256 | Metadata SHA-256 |
|---|---|---|---|
| Unquantized GPU | `032f5264bd1c2c1858b7727af91c0b7adc9e77b69282f54e85a0a81e6a8a8a56` | `e2df45344f4092d46bae98642d049b14c1a2d7bfc20df49fe322d47088faf4fe` | `1696fd6dc6fb64a1cc4fe33e7de39cb555fe68e05f654c57f81ddc3c93464c6a` |
| Ternary GPU | `57c980df2106c2f1800d42aaff78c79758016dfb7549249e97bea66a08c4c73f` | `5584a05dfddf4aa0f179d9f3f2814a885a9c09a117463e3c996e974c8224d308` | `9b973aad835320766c6f3aef907fbb80f335383e69f432ccba7e3ca97ef80107` |

Both evaluators completed 256 rows and hash checks. This is a development
validation comparison, not a blind final evaluation. It completes the Phase 7
initial-damage measurement only. No real Teacher train cache, QAT optimizer
step, trained recovery checkpoint, Recovery LoRA, or final/sealed evaluation
exists yet. Teacher artifacts were read-only; no active Teacher Python process
was present during these GPU evaluations.

## Phase 8 groundwork

The branch now has a CPU-tested option-supervision loss primitive combining
Teacher-to-student option KL, labeled cross entropy, and Brier loss. Its
train-only JSONL cache loader fails closed on non-train inputs, missing or
duplicate samples, mismatched Teacher identity/temperature, reordered options,
invalid distributions, and Teacher logits/probabilities that disagree. This is
only an implementation contract: no real Teacher train cache has been
generated, no recovery optimizer step has run, and Phase 8 QAT is not complete.
Teacher artifacts and corpus files were not modified; the saved Teacher predictions were read-only.


## QAT training performance audit and safe handoff (PR #14, not yet deployed)

The actual QAT trainer is `scripts/train_student_ternary_qat.py` on
`codex/ternary-student-recovery`, not the separate
`feat/ternary-student-recovery` reference CLI. Attempt 05 uses
`gradient_accumulation_steps: 4`, `evaluation_interval: 128`,
`checkpoint_interval: 128`, and `max_steps: 1215`. Between update
steps 128 and 256 it processes another 512 training examples. It does
**not** repeatedly evaluate the validation set inside that interval.

The audited trainer originally fake-quantized its targeted matrix
weights on every model forward. Each example separately forwards the
question and option batch. The PR's CPU-FP32-master-backed update cache
quantizes each *used* target once per four-example optimizer update,
and restores its original BF16 shadow from the existing FP32 CPU master
after all four backward passes. The original STE gradient to those
shadows is preserved. This does not add another full-model GPU weight
cache. Validation uses a separate read-only-to-the-outside CPU-backed
shadow snapshot so ternary values are prepared once per validation and
the exact original shadows/STE are restored even if evaluation raises.
The no-grad validation cache requires about one BF16 model's target
weight bytes of *additional host RAM* during the validation pass.
This is a reference optimization pending actual 16 GiB GPU timings,
not a claimed speedup or memory-fit result.

The PR emits atomic `latest-progress.json` and a flushed JSON status
line after update 1 and every 4 updates, as well as around validation
and selected-shadow saves. A report of GPU saturation alone does not
distinguish a slow update from a slow validation; these timing events do.

**A selected `best-shadow.safetensors` is NOT an optimizer checkpoint.**
The currently running old process cannot pick up a change to this script,
and its state does not contain optimizer/scheduler or RNG snapshots.
Therefore it cannot be resumed *exactly* from step 128 or 256 by
switching code. Do not describe it as a strict resume.

For an explicit optimizer-reset continuation, first stop the source
process after verifying that the selected `best-shadow` and
`run-metadata.json` are consistent and the process has exited.
Use a **different output directory** in a separately reviewed copy of
the config (do not overwrite the original). Then:

```powershell
python scripts/train_student_ternary_qat.py --config CONFIG_WITH_NEW_OUTPUT_DIR.yaml --repo-root . --teacher-repo-root TEACHER_REPO_ROOT --warm-start-run C:/CodexArtifacts/embeddinggemma2-ternary-qat-v0-attempt-05-adafactor
```

The warm start checks selected-step alignment with the 128-step
checkpoint interval, matching model revision, immutable Teacher cache,
data and validation hashes, deterministic sample order, target
inventory, option-readout settings, and the selected shadow SHA-256
before and after loading. It begins from the previously selected
`best_validation_step` and skips precisely those consumed examples
in the same one-pass sample order, maintaining the global cosine
scheduler *position*. Optimizer moments and CPU FP32 master fractional
state are reinitialized from the saved BF16 weights and are **not**
restored. The new run records that fact and its source-run hashes.
If the source reached step 256 but best remained step 128, the
continuation deliberately restarts from the selected step 128; the
intervening updates are discarded. Never switch or stop the live
process without explicit operator confirmation. PR #14 remains
separate and unmerged until checks and real GPU tests pass.
